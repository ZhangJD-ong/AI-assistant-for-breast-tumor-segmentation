import torch
import gc
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from Model.Models import TranRUnet
from torch.utils.data import DataLoader
from utils import util
import os
from dataset.dataset_lits_test import Test_all_Datasets,Recompone_tool

def load(file):
    itkimage = sitk.ReadImage(file)
    image = sitk.GetArrayFromImage(itkimage)
    return image


def data_preprocess(original_data_path = r'Data/Original_data'):

    filename = os.listdir(os.path.join(os.getcwd(),original_data_path))
    for files in filename:
        sub_filename = os.listdir(os.path.join(os.getcwd(),original_data_path,files))
        util.mkdir(os.path.join(os.getcwd(),r'Data/Sampled_data',files))
        for sub_files in sub_filename:
            aa = sitk.ReadImage(os.path.join(os.getcwd(),original_data_path,files,sub_files))
            ss = util.resampleVolume([1,1,1],aa)
            new = util.rm_zeros(ss)
            sitk.WriteImage(new, os.path.join(os.getcwd(),r'Data/Sampled_data',files,sub_files))

    file = open(os.path.join(os.getcwd(),r'Data/Sampled_data/test.txt'), 'w')
    for ii in filename:
        file.write("'"+str(ii)+"'");
        file.write('\n')
    file.close()

def data_postprocess(part,original_data_path = r'Data/Original_data'):
    filename = os.listdir(original_data_path)
    for files in filename:
        a = sitk.ReadImage(os.path.join(os.getcwd(),r'Data/Sampled_data',files,'P2.nii.gz'))
        if part == 'Tumor':
            s = sitk.ReadImage(os.path.join(os.getcwd(),r'Results/Tumor_sampled',files+'.nii.gz'))
        elif part == 'Breast':
            s = sitk.ReadImage(os.path.join(os.getcwd(),r'Results/Breast_sampled', files + '.nii.gz'))
        else:
            print('??? What do you want to do?')
        s.SetOrigin(a.GetOrigin())
        s.SetDirection(a.GetDirection())
        s.SetSpacing(a.GetSpacing())

        sss = sitk.ReadImage(os.path.join(os.getcwd(),original_data_path,files,'P2.nii.gz'))
        ss = util.resize_image_itk(s, sss)
        if part == 'Tumor':
            util.mkdir(os.path.join(os.getcwd(),r'Results/Tumor'))
            sitk.WriteImage(ss,os.path.join(os.getcwd(),r'Results/Tumor',files+'.nii.gz'))
        else:
            util.mkdir(os.path.join(os.getcwd(), r'Results/Breast'))
            sitk.WriteImage(ss, os.path.join(os.getcwd(),r'Results/Breast', files + '.nii.gz'))




def test_all(model_name='best_model.pth'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(device)
    trans_param = {'hidden_size': 768, 'MLP_dim': 2048, 'Num_heads': 12, \
    'Dropout_rate': 0.1, 'Attention_dropout_rate':0.0, 'Trans_num_layers':12}

    model = TranRUnet(1,1,16,trans_param).to(device)
    ckpt = torch.load(os.path.join(os.getcwd(),r'Trained_model',model_name),map_location=device)
    model.load_state_dict(ckpt['model'])

    model.eval()
    cut_param = {'patch_s': 32, 'patch_h': 96, 'patch_w': 96,
                 'stride_s': 16, 'stride_h': 48, 'stride_w': 48}
    datasets = Test_all_Datasets(os.path.join(os.getcwd(),r'Data/Sampled_data'), cut_param)

    for img_dataset,TS_all, original_shape, new_shape,itkimage,file_idx in datasets:
        save_tool = Recompone_tool(original_shape, new_shape, cut_param)
        dataloader = DataLoader(img_dataset, batch_size=16, num_workers=1, shuffle=False)

        with torch.no_grad():
            for x0,x1,x2,x3,x4,x5,gt in tqdm(dataloader):
                x0, x1, x2 = x0.unsqueeze(1).to(device), x1.unsqueeze(1).to(device), x2.unsqueeze(1).to(device)
                x3, x4, x5 = x3.unsqueeze(1).to(device), x4.unsqueeze(1).to(device), x5.unsqueeze(1).to(device)
                TS = torch.from_numpy(np.repeat(TS_all[np.newaxis,:],x0.shape[0],0)).to(device)

                output = model(x0,x1,x2,x3,x4,x5,TS)
                output = (output>=0.5).type(torch.float32)
                save_tool.add_result(output.detach().cpu())

        pred = save_tool.recompone_overlap()

        recon = (pred.numpy()>0.5).astype(np.uint16)#*(mask.astype(np.uint16))
        #recon[-5:,:,:]=0
        #recon[:5,:,:]=0

        Pred = sitk.GetImageFromArray(recon)
        util.mkdir(os.path.join(os.getcwd(),r'Results/Tumor_sampled'))
        sitk.WriteImage(Pred,os.path.join(os.path.join(os.getcwd(),r'Results/Tumor_sampled'), file_idx+'.nii.gz'))
        del pred,recon,Pred,save_tool,gt
        gc.collect()



if __name__ == '__main__':
    data_preprocess()
    test_all('best_model.pth')
    #data_postprocess('Breast')
    data_postprocess('Tumor')       
