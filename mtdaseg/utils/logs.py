import os

import torch


def print_losses(i_iter, losses, cfg):
    ''' Print out the current losses '''

    list_logs = ['[{:6}/{:6}] '.format(i_iter, cfg.TRAIN.EARLY_STOP)]

    for name, loss in losses.items():
        if name == 'ETA':
            list_logs.append('{}: {}'.format(name, loss))
        else:
            list_logs.append('{}: {:.4f}'.format(name, loss))
    
    logs = list_logs[0] + ' || '.join(list_logs[1:])
    save_logs(logs, cfg)
    print(logs)
    
    
def save_logs(logs, cfg):
    ''' Save the logs to the log file '''

    with open(os.path.join(cfg.OUTPUT_DIR.EXP, 'logs.txt'), 'a') as f:
        f.write(logs + '\n')
        
        
def save_model(model, file_path):
    ''' Save the model's weight parameters '''

    torch.save(model.state_dict(), file_path)
    
    
def display_stats(name_classes, inters_over_union_classes, cfg):
    ''' Print out the model performance '''
    
    for ind_class in range(cfg.DATASETS.NUM_CLASSES):
        print('{:15}: {}'.format(name_classes[ind_class],
                                 str(round(inters_over_union_classes[ind_class] * 100, 2))))