# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt

from fcn import SaliencyFCN
from DHF1K_loader import DHF1KDataset

# from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os
import sklearn.metrics as sklrn
from salience_metrics import auc_judd
from salience_metrics import all_metrics
import time
import argparse
import cv2


def draw_graph(values, x_label, y_label , output_dir, graph_name):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(values)
    plt.savefig(os.path.join(output_dir, graph_name + '.png'))
    plt.clf()



def load_checkpoint(args, fcn_model, optimizer, output_dir):
    checkpoint = torch.load(os.path.join(output_dir, 'checkpoints', args.checkpoint))
    fcn_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if checkpoint['epoch_continue']:
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = checkpoint['epoch']
    val_results = checkpoint['val_results']
    epoch_losses = checkpoint['train_loss']
    iter_losses = checkpoint['iter_losses']
    if checkpoint['epoch_continue']:
        start_iter = checkpoint['start_iter'] + 1
    else:
        start_iter = 0 #start new epoch
    print("Training start from {}".format(os.path.join(output_dir, 'checkpoints', args.checkpoint)))
    return epoch_losses, iter_losses, val_results, start_epoch, start_iter

def train(args, train_loader, val_loader, test_loader, fcn_model,
           scheduler, optimizer, output_dir, use_gpu, epochs, criterion):
    
    # Load checkpoint and prepare metric arrays
    if args.checkpoint is not None:
        epoch_losses, iter_losses, val_results, start_epoch, start_iter = \
                                load_checkpoint(args, fcn_model, optimizer, output_dir)
    else:
        epoch_losses = []
        iter_losses = []
        val_results = {'epoch_loss':[], 'auc_judd':[], 'sauc':[], 'cc':[], 'nss':[]}
        start_epoch = 0
        start_iter = 0
        print("New training is starting")
    
    
    # Only Validate
    if args.validate:
        val(args, start_epoch, fcn_model, criterion, val_loader, use_gpu, output_dir)
        return
    
    # Only Test
    if args.test:
        val(args, start_epoch, fcn_model, criterion, test_loader, use_gpu, output_dir)
        return
    
    for epoch in range(start_epoch, epochs):
        fcn_model.train()
        print("####################################################")
        print('Training Epoch: ' + str(epoch))
        scheduler.step()

        buffer_losses = []
        epoch_start = time.time()
        
        # Epoch of Training
        for iter, batch in enumerate(train_loader):
            cur_iter = iter + start_iter
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['smap_Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['smap_Y'])

            # Train one iteration
            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iter_losses.append(loss.item())
            buffer_losses.append(loss.item())
            
            if cur_iter % 10 == 0:
                print("train epoch {}, iter{}/{}, loss: {}, elapsed {}"\
                      .format(epoch, cur_iter, train_loader.__len__(), loss.item(), time.time()-epoch_start))
                draw_graph(iter_losses, "Iteration", "Loss", output_dir, "train_iter_loss")
            
            # Save checkpoint
            if cur_iter % 50 == 0:
                torch.save({'epoch':epoch,
                    'model_state_dict': fcn_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': epoch_losses,
                    'val_results': val_results,
                    'iter_losses': iter_losses, 
                    'epoch_continue': True,
                    'start_iter': cur_iter
                    }, os.path.join(output_dir, 'checkpoints', 'epoch_' + str(epoch) + '_iter_' + str(cur_iter) + '.checkpoint'))
        
        # End of epoch info
        epoch_end = time.time()
        hours, rem = divmod(epoch_end-epoch_start, 3600)
        minutes, seconds = divmod(rem, 60)
           
        epoch_losses.append(np.mean(np.array(buffer_losses)))
        buffer_losses = []
        print("##### Finish epoch {}, time elapsed {}h {}m {}s #####".format(epoch, hours, minutes, seconds))
        print("####################################################")

        # Validate model and Prepare metrics
        val_epoch_results = val(args, epoch, fcn_model, criterion, val_loader, use_gpu, output_dir)
        for key,value in val_epoch_results.items():
            val_results[key].append(value)
        
        # Save epoch checkpoint
        torch.save({'epoch':epoch,
                    'model_state_dict': fcn_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': epoch_losses,
                    'val_results': val_results,
                    'iter_losses': iter_losses,
                    'epoch_continue': False,
                    'start_iter': 0
            }, os.path.join(output_dir, 'checkpoints', 'epoch_' + str(epoch) + '.checkpoint'))
        
        # Draw Graphs
        draw_graph(epoch_losses, "Epoch", "Loss", output_dir, "train_epoch_loss")
        draw_graph(val_results['epoch_loss'], 'Epoch', 'loss', output_dir, 'val_epoch_loss')
        draw_graph(val_results['auc_judd'], 'Epoch', 'auc_judd', output_dir, 'val_auc_judd')
        draw_graph(val_results['sauc'], 'Epoch', 'sauc', output_dir, 'val_sauc')


def val(args, epoch, fcn_model, criterion, val_loader, use_gpu, output_dir):
    
    # Prepare metric arrays and model switch
    fcn_model.eval()
    metrics = ['auc_judd', 'sauc', 'cc', 'nss']
#     metrics = []
    results = {'auc_judd': [], 'sauc': [], 'cc': [], 'nss': []}
    loss_buffer = []
    epoch_scores = {'epoch_loss':0, 'auc_judd': 0, 'sauc': 0, 'cc': 0, 'nss': 0}

    # Epoch start
    epoch_start = time.time()
    for iter, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
            labels = Variable(batch['smap_Y'].cuda())
        else:
            inputs, labels = Variable(batch['X']), Variable(batch['smap_Y'])
            
        # Feedforward a batch
        outputs = fcn_model(inputs)
        loss = criterion(outputs, labels)
        loss_buffer.append(loss.item())
        outputs = outputs.data.cpu().numpy()

        # Validation metric calculation
        target = np.array(batch['fixation_Y'])
        
        if args.visual_result:
            for i in range(outputs.shape[0]):
                estimated = batch['smap_Y'][i].numpy().transpose(1,2,0)
#                 vid_name = batch['name'][i].split('_')[0]
#                 output_smap_dir = os.path.join(output_dir, 'examples', 'estimated', vid_name)
                cv2.imwrite(os.path.join(output_dir, 'examples', 'estimated', 'smap_'+batch['name'][i]+'.jpg'), estimated*255)


                
        start = time.time()
        results = all_metrics(target, outputs, results, metrics)
        end = time.time()
        
        if iter % 1 == 0:
            metric_text = ""
            for metric in metrics:
                metric_text += "{}: {}, ".format(metric, np.mean(np.array(results[metric])))
            print("val epoch {}, elapsed {}, iteration {}/{}, {}".format(epoch, end-start, iter, val_loader.__len__() , metric_text))
  
    epoch_scores['epoch_loss'] = np.mean(np.array(loss_buffer))
    
    metric_text = ""
    for metric in metrics:
        epoch_scores[metric] = np.mean(np.array(results[metric]))
        metric_text += "{}: {}, ".format(metric, epoch_scores[metric])

    epoch_end = time.time()
    hours, rem = divmod(epoch_end-epoch_start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Validation Epoch {}, elapsed {}h {}m {}s, {}".format(epoch,hours, minutes, seconds, metric_text))

    return epoch_scores

def parse_args():
    parser = argparse.ArgumentParser(description='Saliency Training')
    parser.add_argument("--num-gpus", type=int, default=1,
    help="# of GPUs to use for training")
    
    parser.add_argument("-g", "--gpu", type=str, default="1",
    help="run on the gpu")

    parser.add_argument(
        '--exp', '-e',
        dest='exp_name', type=str, default="exp"
    )
    
    parser.add_argument(
        '--exp_dir',
        dest='exp_dir', type=str, default='../experiments')
    
    parser.add_argument(
        '--checkpoint',
        dest='checkpoint', type=str)
    
    parser.add_argument(
        "--batchsize", type=int, default=128,
        dest='batchsize', help="batchsize")
    
    parser.add_argument(
        '--flow', dest='flow', action='store_true',
        help='evaluate model on validation set')
    
    parser.add_argument(
        '--rgb', dest='rgb', action='store_true',
        help='evaluate model on validation set')
    
    parser.add_argument(
        '--validate', dest='validate', action='store_true',
        help='evaluate model on validation set')
    parser.add_argument(
        '--test', dest='test', action='store_true',
        help='evaluate model on test set')
    parser.add_argument(
        '--visual-result', dest='visual_result', action='store_true',
        help='evaluate model on test set')
    
    return parser.parse_args()

def main():
    
    # Configurations
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    n_class    = 1
    batch_size = args.batchsize
    epochs     = 10
    lr         = 1e-4
    momentum   = 0
    w_decay    = 1e-5
    step_size  = 50
    gamma      = 0.5
    configs    = "FCNs-MSE_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
    print("Configs:", configs)
    
    
    # Create dir for model
    output_dir=os.path.join(args.exp_dir, args.exp_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(os.path.join(output_dir, 'checkpoints')):
        os.makedirs(os.path.join(output_dir, 'checkpoints'))
    
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))
    
    if args.flow:
        print("Flow datasets loaded")
        # Traning and validation loaders
        train_data = DHF1KDataset('../dataset/DHF1K/train/flows_fix', 
                                  '../dataset/DHF1K/train/target_fix', 640, 360)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
        
        val_data = DHF1KDataset('../dataset/DHF1K/val/flows_fix',
                                 '../dataset/DHF1K/val/target_fix', 640, 360, small_part = 10)
        val_loader = DataLoader(val_data, batch_size=16, num_workers=1)
        
        test_data = DHF1KDataset('../dataset/DHF1K/test/flows_fix',
                                  '../dataset/DHF1K/test/target', 640, 360)
        test_loader = DataLoader(test_data, batch_size=16, num_workers=1)

    if args.rgb:
        print("RGB datasets loaded")
        # Traning and validation loaders
        train_data = DHF1KDataset('../dataset/DHF1K/train/data_fix', 
                                  '../dataset/DHF1K/train/target_fix', 640, 360)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
        
        val_data = DHF1KDataset('../dataset/DHF1K/val/data',
                                 '../dataset/DHF1K/val/target', 640, 360, small_part = 10)
        val_loader = DataLoader(val_data, batch_size=16, num_workers=1)
        
        test_data = DHF1KDataset('../dataset/DHF1K/test/data',
                                  '../dataset/DHF1K/test/target', 640, 360)
        test_loader = DataLoader(test_data, batch_size=16, num_workers=1)

    fcn_model = SaliencyFCN(n_class=n_class)
    
    if use_gpu:
        ts = time.time()
        fcn_model = fcn_model.cuda()
        fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
        print("Finish cuda loading, time elapsed {}".format(time.time() - ts))
    
    # Optimizer and Loss Function
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs
    
    train(args, train_loader, val_loader, test_loader, fcn_model, scheduler, optimizer, output_dir, use_gpu, epochs, criterion)
    
if __name__ == "__main__":

    main()
  
