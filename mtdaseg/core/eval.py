import numpy as np

import torch
import torch.nn as nn

from mtdaseg.utils.logs import display_stats
from mtdaseg.utils.func import per_class_iu, fast_hist


def eval_model(cfg, model, test_loaders, fixed_test_size=True):
    ''' Evaluate the model performance '''
    
    device = cfg.MODEL.GPU_ID
    interp = nn.Upsample(size=cfg.TEST.INPUT_SIZE[::-1], mode='bilinear', align_corners=True).cuda(device)
    list_miou = []
    list_iou_classes = []
    
    # Evalueate the model performance
    with torch.no_grad():
        for test_loader in test_loaders:
            print('\n[ Evaluating the model performance on {} dataset ]'.format(test_loader.dataset.dataset_name))
            hist = np.zeros((cfg.DATASETS.NUM_CLASSES, cfg.DATASETS.NUM_CLASSES))
            
            for i_iter, batch in enumerate(test_loader):
                image, label, _, _, _ = batch
                if not fixed_test_size:
                    interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
                
                with torch.no_grad():
                    pred_main = model(image.cuda(device))[1]
                    output = interp(pred_main).cpu().data[0].numpy()
                    output = output.transpose(1, 2, 0)
                    output = np.argmax(output, axis=2)
                    
                label = label.numpy()[0]
                hist += fast_hist(label.flatten(), output.flatten(), cfg.DATASETS.NUM_CLASSES)
                print('[{:6}/{:6}] Computing mIoU for {}'.format(i_iter + 1, len(test_loader), test_loader.dataset.dataset_name), end='\r')
                
            # Get IoU per class and mIoU
            iou_classes = per_class_iu(hist)
            miou = round(np.nanmean(iou_classes) * 100, 2)

            list_miou.append(miou)
            list_iou_classes.append(iou_classes)
            
            # Dsiplay the model performance
            print(' ' * 100, end='\r')
            print('{:15}: {}'.format('mIoU', miou))
            display_stats(test_loader.dataset.class_names, iou_classes, cfg)
            
    return list_miou, list_iou_classes
