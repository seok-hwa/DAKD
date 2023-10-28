import os
import copy
import time
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from mtdaseg.core.eval import eval_model
from mtdaseg.utils.loss import cross_entropy_2d
from mtdaseg.utils.logs import print_losses, save_logs, save_model
from mtdaseg.dataset.builder import get_mtda_loader, get_tgt_dataloader
from mtdaseg.utils.func import (
    get_bridges,
    pseudo_labels,
    update_teacher,
    get_bridge_weight,
    adjust_learning_rate,
)


def train_source_only(model, src_loader, cfg):
    ''' Train a model only on the source domain '''

    # Set the information required for training
    device = cfg.MODEL.GPU_ID
    input_size = cfg.INPUT.INPUT_SIZE_SOURCE
    best_model = {'model': copy.deepcopy(model), 'avg': 0., 'i_iter': 0}

    # Segmentation network settings
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # Optimizer setting
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    
    # Interpolator for output segmaps
    interp = nn.Upsample(size=input_size[::-1], mode='bilinear', align_corners=True).cuda(device)

    # Start the training
    start_time = time.time()
    src_loader_iter = enumerate(src_loader)

    for i_iter in range(cfg.TRAIN.INIT_ITER, cfg.TRAIN.EARLY_STOP):
        cfg.TRAIN.CUR_ITER = i_iter

        if (i_iter > 0) and (i_iter % len(src_loader) == 0): 
            src_loader_iter = enumerate(src_loader)

        # Adapt learning rate if needed
        adjust_learning_rate(optimizer, cfg)

        # TRAIN ON SOURCE
        # Get the source data and predict the output segmaps
        _, (src_imgs, src_lbls, _, _, _) = src_loader_iter.__next__()
        src_imgs = src_imgs.cuda(device)
        src_lbls = src_lbls.cuda(device)
        pred_src_aux, pred_src_main = model(src_imgs)

        # Compute auxiliary losses for source data
        if cfg.MODEL.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = cross_entropy_2d(pred_src_aux, src_lbls.long(), torch.ones(src_lbls.shape, device=device), cfg)
        else:
            loss_seg_src_aux = 0

        # Compute main losses for source data
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = cross_entropy_2d(pred_src_main, src_lbls.long(), torch.ones(src_lbls.shape, device=device), cfg)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print out losses
        if (i_iter + 1) % cfg.TRAIN.LOG_STEP == 0:
            eta_seconds = ((time.time() - start_time) / (i_iter - cfg.TRAIN.INIT_ITER + 1)) * (cfg.TRAIN.EARLY_STOP - (i_iter + 1))
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            current_losses = {'loss_seg_src': loss,
                              'ETA': eta_string}
            print_losses(i_iter + 1, current_losses, cfg)

        # Save a snapshot
        if (i_iter + 1) % cfg.TRAIN.SNAPSHOT_STEP == 0:
            print('\nTaking sanpshot ...')
            print('Exp = {}'.format(cfg.OUTPUT_DIR.SNAPSHOT))
            save_logs('\nTaking sanpshot ...\nExp = {}'.format(cfg.OUTPUT_DIR.SNAPSHOT), cfg)

            snapshot_dir = Path(cfg.OUTPUT_DIR.SNAPSHOT)
            save_model(model, snapshot_dir / 'model_{}.pth'.format(i_iter + 1))
            save_model(model, snapshot_dir / 'model_last.pth')

        # Measure the model performance
        if ((i_iter + 1) % cfg.TEST.EVAL_STEP == 0):
            avg = test_model(i_iter + 1, model, cfg)
            
            if avg >= best_model['avg']:
                best_model['model'] = copy.deepcopy(model)
                best_model['avg'] = avg
                best_model['i_iter'] = i_iter
                save_model(best_model['model'], snapshot_dir / 'model_best.pth')

    # Save the last snapshot
    print('\nTaking sanpshot ...')
    print('Exp = {}'.format(cfg.OUTPUT_DIR.SNAPSHOT))
    save_logs('\nTaking sanpshot ...\nExp = {}'.format(cfg.OUTPUT_DIR.SNAPSHOT), cfg)
    save_model(model, os.path.join(cfg.OUTPUT_DIR.SNAPSHOT, 'model_last.pth'))

    # Print out the best performance
    print('\nPrint out the best performance...')
    save_logs('\nPrint out the best performance...', cfg)
    with open(os.path.join(cfg.OUTPUT_DIR.EXP, 'performance.txt'), 'a') as f:
        f.write('Best Performance:\n')
    test_model(best_model['i_iter'] + 1, best_model['model'], cfg)


def train_baseline(model, src_loader, tgt_loaders, cfg):
    ''' Train the baseline model on the source domain and the domain bridges '''

    # Set the information required for training
    device = cfg.MODEL.GPU_ID
    input_size_source = cfg.INPUT.INPUT_SIZE_SOURCE
    input_size_target = cfg.INPUT.INPUT_SIZE_TARGET
    best_model = {'model': copy.deepcopy(model), 'avg': 0., 'i_iter': 0}
    
    # Setup the segmentation networks
    model.train()
    model.to(device)
    
    teacher = copy.deepcopy(model)
    teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    cudnn.benchmark = True
    cudnn.enabled = True
    
    # Optimizer setting
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    
    # Interpolator for output segmaps
    assert input_size_source == input_size_target, 'The input sizes of source and target domain must be same.'
    interp_src = nn.Upsample(size=input_size_source[::-1], mode='bilinear', align_corners=True).cuda(device)
    interp_tgt = nn.Upsample(size=input_size_target[::-1], mode='bilinear', align_corners=True).cuda(device)
    
    # Start the training
    start_time = time.time()
    src_loader_iter = enumerate(src_loader)
    tgt_loader, tgt_loader_iter, prev_iter = get_tgt_dataloader(tgt_loader=None, tgt_loaders=tgt_loaders, cfg=cfg)

    for i_iter in range(cfg.TRAIN.INIT_ITER, cfg.TRAIN.EARLY_STOP):
        cfg.TRAIN.CUR_ITER = i_iter
        
        if (i_iter > 0) and (i_iter % len(src_loader) == 0): 
            src_loader_iter = enumerate(src_loader)
        if (i_iter > 0) and ((i_iter - prev_iter) % len(tgt_loader) == 0):
            tgt_loader, tgt_loader_iter, prev_iter = get_tgt_dataloader(tgt_loader=tgt_loader, tgt_loaders=tgt_loaders, cfg=cfg)

        # Reset an optimizer
        optimizer.zero_grad()
        
        # Adapt learning rate if needed
        adjust_learning_rate(optimizer, cfg)
        
        # TRAIN ON SOURCE
        # Get the source data and predict the output segmaps
        _, (src_imgs, src_lbls, _, _, _) = src_loader_iter.__next__()
        src_imgs = src_imgs.cuda(device)
        src_lbls = src_lbls.cuda(device)
        pred_src_aux, pred_src_main = model(src_imgs)
        
        # Compute auxiliary losses for source data
        if cfg.MODEL.MULTI_LEVEL:
            pred_src_aux = interp_src(pred_src_aux)
            loss_seg_src_aux = cross_entropy_2d(pred_src_aux, src_lbls.long(), torch.ones(src_lbls.shape, device=device), cfg)
        else:
            loss_seg_src_aux = 0
    
        # Compute main losses for source data
        pred_src_main = interp_src(pred_src_main)
        loss_seg_src_main = cross_entropy_2d(pred_src_main, src_lbls.long(), torch.ones(src_lbls.shape, device=device), cfg)
        src_loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                   + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        src_loss.backward()
        
        # TRAIN ON BRIDGES
        # Get the target data and pseudo labels
        _, (tgt_imgs, _, _, _, _) = tgt_loader_iter.__next__()
        tgt_imgs = tgt_imgs.cuda(device)
        tgt_lbls, tgt_preds, _ = pseudo_labels(teacher, tgt_imgs, interp_tgt)
        
        # Make the bridges and weight for bridge losses
        brg_imgs, brg_lbls, masks = get_bridges(src_imgs, src_lbls, 
                                                tgt_imgs, tgt_lbls, 
                                                cfg.DA.BRIDGE_TYPE, cfg,
                                                cs_type='random')
        brg_weights = get_bridge_weight(masks, tgt_preds, cfg)
    
        # predict the output segmaps for bridges
        pred_brg_aux, pred_brg_main = model(brg_imgs)
        
        # Compute auxiliary losses for bridges
        if cfg.MODEL.MULTI_LEVEL:
            pred_brg_aux = interp_tgt(pred_brg_aux)
            loss_seg_brg_aux = cross_entropy_2d(pred_brg_aux, brg_lbls.long(), brg_weights, cfg)
        else:
            loss_seg_brg_aux = 0
            
        # Compute main losses for bridges
        pred_brg_main = interp_tgt(pred_brg_main)
        loss_seg_brg_main = cross_entropy_2d(pred_brg_main, brg_lbls.long(), brg_weights, cfg)
        brg_loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_brg_main
                    + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_brg_aux)
        brg_loss = brg_loss * cfg.TRAIN.LAMBDA_BRIDGE
        brg_loss.backward()
        optimizer.step()

        # Update the teacher network through EMA
        if i_iter % cfg.TRAIN.EMA_STEP == 0:
            update_teacher(teacher, model, cfg)
        
        # Print out losses
        if (i_iter + 1) % cfg.TRAIN.LOG_STEP == 0:
            eta_seconds = ((time.time() - start_time) / (i_iter - cfg.TRAIN.INIT_ITER + 1)) * (cfg.TRAIN.EARLY_STOP - (i_iter + 1))
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            current_losses = {'loss_seg_src': src_loss,
                              'loss_seg_brg': brg_loss,
                              'ETA': eta_string}
            print_losses(i_iter + 1, current_losses, cfg)
            
        # Save a snapshot
        if (i_iter + 1) % cfg.TRAIN.SNAPSHOT_STEP == 0:
            print('\nTaking sanpshot ...')
            print('Exp = {}'.format(cfg.OUTPUT_DIR.SNAPSHOT))
            save_logs('\nTaking sanpshot ...\nExp = {}'.format(cfg.OUTPUT_DIR.SNAPSHOT), cfg)
 
            snapshot_dir = Path(cfg.OUTPUT_DIR.SNAPSHOT)
            save_model(model, snapshot_dir / 'model_{}.pth'.format(i_iter + 1))
            save_model(model, snapshot_dir / 'model_last.pth')
            
        # Measure the model performance
        if ((i_iter + 1) % cfg.TEST.EVAL_STEP == 0):
            avg = test_model(i_iter + 1, model, cfg)
            
            if avg >= best_model['avg']:
                best_model['model'] = copy.deepcopy(model)
                best_model['avg'] = avg
                best_model['i_iter'] = i_iter
                save_model(best_model['model'], snapshot_dir / 'model_best.pth')
        
    # Save the last snapshot
    print('\nTaking sanpshot ...')
    print('Exp = {}'.format(cfg.OUTPUT_DIR.SNAPSHOT))
    save_logs('\nTaking sanpshot ...\nExp = {}'.format(cfg.OUTPUT_DIR.SNAPSHOT), cfg)
    save_model(model, os.path.join(cfg.OUTPUT_DIR.SNAPSHOT, 'model_last.pth'))

    # Print out the best performance
    print('\nPrint out the best performance...')
    save_logs('\nPrint out the best performance...', cfg)
    with open(os.path.join(cfg.OUTPUT_DIR.EXP, 'performance.txt'), 'a') as f:
        f.write('Best Performance:\n')
    test_model(best_model['i_iter'] + 1, best_model['model'], cfg)


def test_model(i_iter, model, cfg):
    ''' Measure the model performance '''

    # Get the names of target datasets
    targets = cfg.DATASETS.TARGET

    # Get the target dataloaders
    test_loaders = get_mtda_loader(cfg, train=False)[1]
        
    # Get the model performance
    model.eval()
    list_miou, list_iou_classes = eval_model(cfg, model, test_loaders)
    
    # Save the model performance
    perform = ['[{:6}/{:6}] '.format(i_iter, cfg.TRAIN.EARLY_STOP),
               'Avg.: {:6.02f}'.format(round(sum(list_miou) / len(list_miou), 2))]
    
    for name, miou in zip(targets, list_miou):
        perform.append('{}: {:6.02f}'.format(name, miou))
    
    with open(os.path.join(cfg.OUTPUT_DIR.EXP, 'performance.txt'), 'a') as f:
        perform = perform[0] + ' || '.join(perform[1:]) + '\n'
        f.write(perform)

    # Save the model performance to the log file
    for test_loader, miou, iou_classes in zip(test_loaders, list_miou, list_iou_classes):
        save_logs('\n[ Evaluating the model performance on {} dataset ]'.format(test_loader.dataset.dataset_name), cfg)
        save_logs('{:15}: {}'.format('mIoU', miou), cfg)

        for ind_class in range(cfg.DATASETS.NUM_CLASSES):
            save_logs('{:15}: {}'.format(test_loader.dataset.class_names[ind_class],
                      str(round(iou_classes[ind_class] * 100, 2))), cfg)

    model.train()
    return sum(list_miou) / len(list_miou)
    

def train_domain_adaptation(model, src_loader, tgt_loader, cfg):
    ''' Train the model through domain adaptation '''

    assert cfg.DA.DA_TYPE in ['SourceOnly', 'DA'], f'Not supported DA type: {cfg.DA.DA_TYPE}'

    if cfg.DA.DA_TYPE == 'SourceOnly':
        train_source_only(model, src_loader, cfg)
    elif cfg.DA.DA_TYPE == 'DA':
        if cfg.DA.DA_METHOD == 'Baseline':
            train_baseline(model, src_loader, tgt_loader, cfg)
        else:
            NotImplementedError(f"Not yet supported DA method: {cfg.TRAIN.DA_METHOD}")
