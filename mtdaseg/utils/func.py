import numpy as np

import torch
import torch.nn.functional as F

def update_teacher(teacher, student, cfg):
    ''' Update the weight parameters of teacher network through EMA '''

    alpha = min(1 - 1 / (cfg.TRAIN.CUR_ITER + 1), cfg.TRAIN.KEEP_RATIO)

    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        if not s_param.data.shape:
            t_param.data = alpha * t_param.data + (1 - alpha) * s_param.data
        else:
            t_param.data[:] = alpha * t_param[:].data[:] + (1 - alpha) * s_param[:].data[:]


def lr_poly(base_lr, iter, max_iter, power):
    ''' Poly_LR scheduler '''

    return base_lr * ((1 - float(iter) / max_iter) ** power)


def adjust_learning_rate(optimizer, cfg):
    ''' Adject learning rate for main segnet '''

    lr = lr_poly(cfg.TRAIN.LEARNING_RATE, cfg.TRAIN.CUR_ITER, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.POWER)
    optimizer.param_groups[0]['lr'] = lr

    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
        

def fast_hist(a, b, n):
    ''' Calculate the fast histogram '''
    
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    ''' Calculate IoU per class '''
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def prob_2_entropy(prob):
    ''' Convert probabilistic prediction maps to weighted self-information maps '''
    
    _, c, _, _ = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def pseudo_labels(model, images, interp=None):
    ''' Return the pseudo labels and entropy maps for input images '''
    
    assert images.ndim == 4, 'The shape of input images must be \'(Batch, Channel, Height, Width)\'.'
    
    with torch.no_grad():
        # Predict the output segmaps
        output = model(images)[1]
        output = interp(output) if interp else output
        soft_output = F.softmax(output)
        
        # Get the entropy maps
        ent_maps = prob_2_entropy(soft_output)
        ent_maps = torch.sum(ent_maps, dim=1)
        
        # Get the pseudo labels
        pseudos = torch.argmax(soft_output, dim=1)
        
    return pseudos, output, ent_maps


def get_bridges(src_imgs, src_lbls, tgt_imgs, tgt_lbls, mode, cfg, **kwargs):
    ''' Make the bridges through mixing strategy '''

    # Make the masks for mixing data
    # The shape of masks is [N, 1, H, W]
    if mode == 'cut':
        masks = get_cut_masks(src_lbls, cfg)
    elif mode == 'class':
        masks = get_class_masks(src_lbls, cfg, **kwargs)
    else:
        raise NotImplementedError(f"Not yet supported mixing strategy: {mode}")
    
    img_masks = masks
    lbl_masks = masks.squeeze(1)
    
    # Make the mixed images and mixed labels
    mixed_imgs = img_masks * src_imgs + (1 - img_masks) * tgt_imgs
    mixed_lbls = lbl_masks * src_lbls + (1 - lbl_masks) * tgt_lbls
    
    return mixed_imgs, mixed_lbls, img_masks


def get_classes(src_lbl, class_ratio, cfg, **kwargs):
    ''' Get the classes to paste into the target data '''

    classes = torch.unique(src_lbl).long()
    classes = classes[classes != cfg.INPUT.IGNORE_LABEL]
    nclasses = classes.shape[0]

    if kwargs['cs_type'] == 'random':
        class_idx = torch.randperm(nclasses, device=classes.device)[:round(nclasses * class_ratio)]
        classes = torch.take(classes, class_idx)

    else:
        raise NotImplementedError(f"Not yet supported selection type: {kwargs['cs_type']}")
    
    return classes


def get_class_masks(src_lbls, cfg, **kwargs):
    ''' Create masks for source data through ClassMix '''

    masks = []

    for src_lbl in src_lbls:
        classes = get_classes(src_lbl, 0.5, cfg, **kwargs)
        masks.append(torch.isin(src_lbl, classes).long().unsqueeze(0))

    masks = torch.stack(masks).to(cfg.MODEL.GPU_ID)
    return masks


def get_cut_masks(labels, cfg):
    ''' Create masks for source data through CutMix '''
    
    n, h, w = labels.shape
    mask_props = 0.4
    
    # Get the sizes of the boxes
    y_props = torch.exp(torch.rand(n, 1) * np.log(mask_props))
    x_props = mask_props / y_props
    sizes = (torch.concat([y_props, x_props], dim=1) * torch.tensor([h, w])[None, :]).round()
    
    # Get the positions of the boxes
    positions = ((torch.tensor([h, w]) - sizes) * torch.rand(sizes.shape)).round()
    rectangles = torch.concat([positions, positions + sizes], dim=1)
    
    # Create the CutMix masks
    masks = torch.zeros((n, 1, h, w), device=cfg.MODEL.GPU_ID)
    for i, (y0, x0, y1, x1) in enumerate(rectangles):
        masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1
        
    return masks


def get_bridge_weight(masks, logits, cfg):
    ''' Get the confidence-based bridge weights for reducing noises '''
    
    probs = F.softmax(logits)
    pseudo_prob, pseudo_label = torch.max(probs, dim=1)
    ps_large_p = pseudo_prob.ge(cfg.TRAIN.CONF_THRESHOLD).long() == 1
    
    ps_size = pseudo_label.nelement()
    pseudo_weight = torch.sum(ps_large_p).item() / ps_size
    pseudo_weight = pseudo_weight * torch.ones(pseudo_prob.shape, device=logits.device)
    gt_weight = torch.ones(pseudo_weight.shape, device=logits.device)
    
    # Create the confidence-based bridge weights
    brg_weight = masks.squeeze(1) * gt_weight + (1 - masks.squeeze(1)) * pseudo_weight
    return brg_weight