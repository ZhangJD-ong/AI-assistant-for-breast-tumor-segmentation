import torch
import numpy as np
import torch.optim as optim
import SimpleITK as sitk
from options.Options import Options_x
from dataset.dataset_lits_train import Lits_DataSet
from Model.Models import TranRUnet
from torch.utils.data import DataLoader
from utils.common import adjust_learning_rate
from utils import logger,util
import torch.nn as nn
from utils.metrics import LossAverage, DiceLoss
import os
from test import test_all
from collections import OrderedDict


def train (train_dataloader,epoch):
    print("=======Epoch:{}======Learning_rate:{}=========".format(epoch,optimizer.param_groups[0]['lr']))

    Loss = LossAverage()
    DICE_Loss = LossAverage()
    BCE_Loss = LossAverage()

    model.train()
    
   
    for i, (x0,x1,x2,x3,x4,x5,gt,T_all,T2,T3,T4) in enumerate(train_dataloader):  # inner loop within one epoch
        ##main model update param
        x0,x1,x2,x3,x4,x5,gt = x0.to(device),x1.to(device),x2.to(device),x3.to(device),x4.to(device),x5.to(device),gt.to(device)
        T_all,T2,T3,T4 = T_all.to(device),T2.to(device),T3.to(device),T4.to(device)
        for t in range(1):
            if t==0:
                TS = T_all
            elif t ==1:
                TS = T2
            elif t ==2:
                TS = T3
            else:
                TS = T4

            pred = model(x0,x1,x2,x3,x4,x5,TS)

            Dice_loss = dice_loss(pred,gt)
            Bce_loss = bce_loss(pred,gt)
            loss = Bce_loss+Dice_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adjust_learning_rate(optimizer,epoch,opt)

            Loss.update(loss.item(),1)
            DICE_Loss.update(Dice_loss.item(),1)
            BCE_Loss.update(Bce_loss.item(), 1)
        
    return OrderedDict({'Loss': Loss.avg,'DICE_Loss':DICE_Loss.avg,'BCE_Loss':BCE_Loss.avg})

    
if __name__ == '__main__':
    opt = Options_x().parse()   # get training options
    device = torch.device('cuda:'+opt.gpu_ids if torch.cuda.is_available() else "cpu")
    print(device)
    trans_param = {'hidden_size': 768, 'MLP_dim': 2048, 'Num_heads': 12, \
    'Dropout_rate': 0.1, 'Attention_dropout_rate':0.0, 'Trans_num_layers':12}

    model = TranRUnet(1,1,16,trans_param).to(device)

    save_path = opt.checkpoints_dir
    dice_loss = DiceLoss()
    bce_loss = torch.nn.BCELoss()
        
    save_result_path = os.path.join(save_path,opt.task_name)
    util.mkdir(save_result_path)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr,weight_decay=1e-5)

    model_save_path = os.path.join(save_result_path,'model')
    util.mkdir(model_save_path)
    logger_save_path = os.path.join(save_result_path,'logger')
    util.mkdir(logger_save_path)
    log_train = logger.Train_Logger(logger_save_path,"train_log")

    train_dataset = Lits_DataSet(opt.datapath, opt.patch_size)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, \
                                  num_workers=opt.num_threads, shuffle=True)

    for epoch in range(opt.epoch):
        epoch = epoch +1
        train_log= train (train_dataloader,epoch)
        log_train.update(epoch,train_log)
        
        state = {'model': model.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(model_save_path, 'latest_model.pth'))

        if epoch%opt.model_save_fre ==0:
            torch.save(state, os.path.join(model_save_path, 'model_'+np.str(epoch)+'.pth'))

        torch.cuda.empty_cache() 


 
        

            
            

            
            
            
            
            
            
