import torch
import gc
import numpy as np
import torch.optim as optim
import SimpleITK as sitk
from options.Options import Options_x
from tqdm import tqdm
from Model.Models import TranRUnet
from torch.utils.data import DataLoader
from utils import logger,util
import time
from utils.metrics import seg_metric
import torch.nn as nn
import os
from dataset.dataset_lits_test import Test_all_Datasets,Recompone_tool
from collections import OrderedDict

def load(file):
    itkimage = sitk.ReadImage(file)
    image = sitk.GetArrayFromImage(itkimage)
    return image


def test_all(model_name='model_200.pth'):
    opt = Options_x().parse()  # get training options
    device = torch.device('cuda:' + opt.gpu_ids if torch.cuda.is_available() else "cpu")
    print(device)
    trans_param = {'hidden_size': 768, 'MLP_dim': 2048, 'Num_heads': 12, \
    'Dropout_rate': 0.1, 'Attention_dropout_rate':0.0, 'Trans_num_layers':12}

    model = TranRUnet(1,1,16,trans_param).to(device)
    ckpt = torch.load(opt.checkpoints_dir + '/' + opt.task_name + '/model/' + model_name,map_location=device)
    model.load_state_dict(ckpt['model'])

    save_result_path = os.path.join(opt.checkpoints_dir, opt.task_name, 'results')
    util.mkdir(save_result_path)
    model.eval()
    log_test = logger.Test_Logger(save_result_path,"train_results")
    cut_param = {'patch_s': opt.patch_size[0], 'patch_h': opt.patch_size[1], 'patch_w': opt.patch_size[2],
                 'stride_s': opt.patch_stride[0], 'stride_h': opt.patch_stride[1], 'stride_w': opt.patch_stride[2]}
    datasets = Test_all_Datasets(opt.datapath, cut_param)

    for img_dataset,TS_all, original_shape, new_shape,itkimage,file_idx,Breast_mask,source in datasets:
        save_tool = Recompone_tool(original_shape, new_shape, cut_param)
        dataloader = DataLoader(img_dataset, batch_size=opt.test_batch, num_workers=opt.num_threads, shuffle=False)

        with torch.no_grad():
            time_start = time.time()
            for x0,x1,x2,x3,x4,x5,gt in tqdm(dataloader):
                x0, x1, x2 = x0.unsqueeze(1).to(device), x1.unsqueeze(1).to(device), x2.unsqueeze(1).to(device)
                x3, x4, x5 = x3.unsqueeze(1).to(device), x4.unsqueeze(1).to(device), x5.unsqueeze(1).to(device)
                TS = torch.from_numpy(np.repeat(TS_all[np.newaxis,:],x0.shape[0],0)).to(device)

                output = model(x0,x1,x2,x3,x4,x5,TS)
                output = (output>=0.5).type(torch.float32)
                save_tool.add_result(output.detach().cpu())

        pred = save_tool.recompone_overlap()

        recon = (pred.numpy()>0.5).astype(np.uint16)*Breast_mask.astype(np.uint16)
        time_end = time.time()
        gt = load(os.path.join(opt.datapath,source+'_gt',file_idx+'.nii.gz'))

        DSC, PPV, SEN, HD = seg_metric(recon,gt,itkimage)
        index_results = OrderedDict({'DSC': DSC,'PPV': PPV,'SEN': SEN,'HD': HD,'TIME': time_end - time_start})
        log_test.update(file_idx,index_results)
        Pred = sitk.GetImageFromArray(np.array(recon))
        result_save_path = os.path.join(save_result_path, file_idx)
        util.mkdir(result_save_path)
        sitk.WriteImage(Pred,os.path.join(result_save_path,'pred.nii.gz'))
        del pred,recon,Pred,save_tool,gt
        gc.collect()


if __name__ == '__main__':
    test_all('latest_model.pth')
                
