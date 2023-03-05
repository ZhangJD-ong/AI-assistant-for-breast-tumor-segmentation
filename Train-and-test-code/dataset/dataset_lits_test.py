import numpy as np
import torch, os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import random
import SimpleITK as sitk

def min_max_normalization(img):
    out=(img - np.min(img))/(np.max(img) - np.min(img) + 0.000001 )
    return out


def normalization( img, lmin=1, rmax=None, dividend=None, quantile=1):
    newimg = img.copy()
    newimg = newimg.astype(np.float32)
    if quantile is not None:
        maxval = round(np.percentile(newimg, 100 - quantile))
        minval = round(np.percentile(newimg, quantile))
        newimg[newimg >= maxval] = maxval
        newimg[newimg <= minval] = minval

    if lmin is not None:
        newimg[newimg < lmin] = lmin
    if rmax is not None:
        newimg[newimg > rmax] = rmax

    minval = np.min(newimg)
    if dividend is None:
        maxval = np.max(newimg)
        newimg = (np.asarray(newimg).astype(np.float32) - minval) / (maxval - minval)
    else:
        newimg = (np.asarray(newimg).astype(np.float32) - minval) / dividend
    return newimg, minval, maxval


def normalization_fix(img, minval, maxval, lmin=1):
    newimg = img.copy()
    newimg = newimg.astype(np.float32)
    if lmin is not None:
        newimg[newimg < lmin] = lmin

    newimg = (np.asarray(newimg).astype(np.float32) - minval) / (maxval - minval)
    return newimg



def load(file):
    itkimage = sitk.ReadImage(file)
    image = sitk.GetArrayFromImage(itkimage)
    return image

def random_crop_3d(pre, pos, sub, gt, crop_size):
    cor_box = maskcor_extract_3d(gt)
    random_x_min, random_x_max = max(cor_box[0, 1] - crop_size[0], 0), min(cor_box[0, 0], pre.shape[0] - crop_size[0])
    random_y_min, random_y_max = max(cor_box[1, 1] - crop_size[1], 0), min(cor_box[1, 0], pre.shape[1] - crop_size[1])
    random_z_min, random_z_max = max(cor_box[2, 1] - crop_size[2], 0), min(cor_box[2, 0], pre.shape[2] - crop_size[2])
    if random_x_min > random_x_max:
        random_x_min, random_x_max = cor_box[0, 0], cor_box[0, 1] - crop_size[0]
    if random_y_min > random_y_max:
        random_y_min, random_y_max = cor_box[1, 0], cor_box[1, 1] - crop_size[1]
    if random_z_min > random_z_max:
        random_z_min, random_z_max = cor_box[2, 0], cor_box[2, 1] - crop_size[2]

    #print(cor_box[0, 0], cor_box[0, 1],cor_box[1, 0], cor_box[1, 1],cor_box[2, 0], cor_box[2, 1])
    #print(random_x_min, random_x_max,random_y_min, random_y_max,random_z_min, random_z_max)
    x_random = random.randint(random_x_min, random_x_max)
    y_random = random.randint(random_y_min, random_y_max)
    z_random = random.randint(random_z_min, random_z_max)

    pre_patch = pre[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                z_random:z_random + crop_size[2]]
    pos_patch = pos[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                z_random:z_random + crop_size[2]]
    sub_patch = sub[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                z_random:z_random + crop_size[2]]
    gt_patch = gt[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
               z_random:z_random + crop_size[2]]

    return pre_patch, pos_patch, sub_patch, gt_patch


def maskcor_extract_3d(mask, padding=(5, 5, 5)):
    # mask_s = mask.shape
    p = np.where(mask > 0)
    a = np.zeros([3, 2], dtype=np.int)
    for i in range(3):
        s = p[i].min()
        e = p[i].max() + 1

        ss = s - padding[i]
        ee = e + padding[i]
        if ss < 0:
            ss = 0
        if ee > mask.shape[i]:
            ee = mask.shape[i]

        a[i, 0] = ss
        a[i, 1] = ee
    return a

class Img_DataSet(Dataset):
    def __init__(self, x0,x1,x2,x3,x4,x5,gt, cut_param):
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.x5 = x5
        self.gt = gt
        

        self.ori_shape = self.x0.shape
        self.cut_param = cut_param

        self.x0 = self.padding_img(self.x0, self.cut_param)
        self.x0 = self.extract_ordered_overlap(self.x0, self.cut_param)
        self.x1 = self.padding_img(self.x1, self.cut_param)
        self.x1 = self.extract_ordered_overlap(self.x1, self.cut_param)
        self.x2 = self.padding_img(self.x2, self.cut_param)
        self.x2 = self.extract_ordered_overlap(self.x2, self.cut_param)
        self.x3 = self.padding_img(self.x3, self.cut_param)
        self.x3 = self.extract_ordered_overlap(self.x3, self.cut_param)
        self.x4 = self.padding_img(self.x4, self.cut_param)
        self.x4 = self.extract_ordered_overlap(self.x4, self.cut_param)
        self.x5 = self.padding_img(self.x5, self.cut_param)
        self.x5 = self.extract_ordered_overlap(self.x5, self.cut_param)

        self.gt = self.padding_img(self.gt, self.cut_param)
        self.gt = self.extract_ordered_overlap(self.gt, self.cut_param)
        
        self.new_shape = self.x0.shape
        
    def __getitem__(self, index):
        x0 = self.x0[index]
        x1 = self.x1[index]
        x2 = self.x2[index]
        x3 = self.x3[index]
        x4 = self.x4[index]
        x5 = self.x5[index]
        gt = self.gt[index]
        
        return torch.from_numpy(x0).type(torch.float32),torch.from_numpy(x1).type(torch.float32),torch.from_numpy(x2).type(torch.float32),\
               torch.from_numpy(x3).type(torch.float32),torch.from_numpy(x4).type(torch.float32),torch.from_numpy(x5).type(torch.float32),\
               torch.from_numpy(gt)
    
    def __len__(self):
        return len(self.x0)


    def padding_img(self, img, C):
        assert (len(img.shape) == 3)  # 3D array
        img_s, img_h, img_w = img.shape
        leftover_s = (img_s - C['patch_s']) % C['stride_s']
        leftover_h = (img_h - C['patch_h']) % C['stride_h']
        leftover_w = (img_w - C['patch_w']) % C['stride_w']
        if (leftover_s != 0):
            s = img_s + (C['stride_s'] - leftover_s)
        else:
            s = img_s

        if (leftover_h != 0):
            h = img_h + (C['stride_h'] - leftover_h)
        else:
            h = img_h

        if (leftover_w != 0):
            w = img_w + (C['stride_w'] - leftover_w)
        else:
            w = img_w

        tmp_full_imgs = np.zeros((s, h, w))
        tmp_full_imgs[:img_s, :img_h, 0:img_w] = img
        #print("Padded images shape: " + str(tmp_full_imgs.shape))
        return tmp_full_imgs

    # Divide all the full_imgs in pacthes
    def extract_ordered_overlap(self, img, C):
        assert (len(img.shape) == 3)  # 3D arrays
        img_s, img_h, img_w = img.shape
        assert ((img_h - C['patch_h']) % C['stride_h'] == 0
                and (img_w - C['patch_w']) % C['stride_w'] == 0
                and (img_s - C['patch_s']) % C['stride_s'] == 0)
        N_patches_s = (img_s - C['patch_s']) // C['stride_s'] + 1
        N_patches_h = (img_h - C['patch_h']) // C['stride_h'] + 1
        N_patches_w = (img_w - C['patch_w']) // C['stride_w'] + 1
        N_patches_img = N_patches_s * N_patches_h * N_patches_w
#        print("Patches number of the image:{} [s={} | h={} | w={}]"\
#               .format(N_patches_img, N_patches_s, N_patches_h, N_patches_w))
        patches = np.empty((N_patches_img, C['patch_s'], C['patch_h'], C['patch_w']))
        iter_tot = 0  # iter over the total number of patches (N_patches)
        for s in range(N_patches_s):  # loop over the full images
            for h in range(N_patches_h):
                for w in range(N_patches_w):
                    patch = img[s * C['stride_s'] : s * C['stride_s']+C['patch_s'],
                            h * C['stride_h']: h * C['stride_h']+C['patch_h'],
                            w * C['stride_w']: w * C['stride_w']+C['patch_w']]
                   
                    patches[iter_tot] = patch
                    iter_tot += 1  # total
        assert (iter_tot == N_patches_img)
        return patches  # array with all the full_imgs divided in patches


class Recompone_tool():
    def __init__(self, img_ori_shape, img_new_shape, Cut_para):
        self.result = None
        self.ori_shape = img_ori_shape
        self.new_shape = img_new_shape
        self.C = Cut_para

    def add_result(self, tensor):
        # tensor = tensor.detach().cpu() # shape: [N,class,s,h,w]
        # tensor_np = np.squeeze(tensor_np,axis=0)
        if self.result is not None:
            self.result = torch.cat((self.result, tensor), dim=0)
        else:
            self.result = tensor

    def recompone_overlap(self):
        """
        :param preds: output of model  shape：[N_patchs_img,3,patch_s,patch_h,patch_w]
        :return: result of recompone output shape: [3,img_s,img_h,img_w]
        """
        patch_s = self.result.shape[2]
        patch_h = self.result.shape[3]
        patch_w = self.result.shape[4]
        N_patches_s = (self.new_shape[0] - patch_s) // self.C['stride_s'] + 1
        N_patches_h = (self.new_shape[1] - patch_h) // self.C['stride_h'] + 1
        N_patches_w = (self.new_shape[2] - patch_w) // self.C['stride_w'] + 1

        N_patches_img = N_patches_s * N_patches_h * N_patches_w
        #print("N_patches_s/h/w:", N_patches_s, N_patches_h, N_patches_w)
        #print("N_patches_img: " + str(N_patches_img))
        assert (self.result.shape[0] == N_patches_img)

        full_prob = torch.zeros(( self.new_shape[0], self.new_shape[1],self.new_shape[2]))  # itialize to zero mega array with sum of Probabilities
        full_sum = torch.zeros((self.new_shape[0], self.new_shape[1], self.new_shape[2]))
        k = 0  # iterator over all the patches
        for s in range(N_patches_s):
            for h in range(N_patches_h):
                for w in range(N_patches_w):
                    #print(k,self.result[k].squeeze().sum())
                    full_prob[s * self.C['stride_s']:s * self.C['stride_s'] + patch_s,
                                 h * self.C['stride_h']:h  * self.C['stride_h'] + patch_h,
                                 w * self.C['stride_w']:w * self.C['stride_w'] + patch_w] += self.result[k].squeeze()
                    full_sum[s * self.C['stride_s']:s * self.C['stride_s'] + patch_s,
                                h * self.C['stride_h']:h * self.C['stride_h'] + patch_h,
                                w * self.C['stride_w']:w * self.C['stride_w'] + patch_w] += 1
                    k += 1
        assert (k == self.result.size(0))
        assert (torch.min(full_sum) >= 1.0)  # at least one
        final_avg = full_prob / full_sum
        # print(final_avg.size())
        img = final_avg[:self.ori_shape[0], :self.ori_shape[1], :self.ori_shape[2]]
        return img

def cal_newshape( img, C):
    assert (len(img.shape) == 3)  # 3D array
    img_s, img_h, img_w = img.shape
    leftover_s = (img_s - C['patch_s']) % C['stride_s']
    leftover_h = (img_h - C['patch_h']) % C['stride_h']
    leftover_w = (img_w - C['patch_w']) % C['stride_w']
    if (leftover_s != 0):
        s = img_s + (C['stride_s'] - leftover_s)
    else:
        s = img_s

    if (leftover_h != 0):
        h = img_h + (C['stride_h'] - leftover_h)
    else:
        h = img_h

    if (leftover_w != 0):
        w = img_w + (C['stride_w'] - leftover_w)
    else:
        w = img_w

    return np.zeros((s, h, w)).shape

def package_torch(pre_patch, pos_patch, sub_patch, gt_patch):
    pre_patch = torch.from_numpy(pre_patch[np.newaxis,np.newaxis,:])
    pos_patch = torch.from_numpy(pos_patch[np.newaxis,np.newaxis,:])
    sub_patch = torch.from_numpy(sub_patch[np.newaxis,np.newaxis,:])
    gt_patch = torch.from_numpy(gt_patch[np.newaxis,np.newaxis,:])
    return pre_patch, pos_patch, sub_patch, gt_patch

def Test_Datasets(dataset_path,size, test_folder = 1):
    f = open(os.path.join(dataset_path,'data_folder', 'test'+str(test_folder)+'.txt'))
    data_list = f.read().splitlines()
    print("The number of test samples is: ", len(data_list))
    for file in data_list:
        #file = str(int(file))
        #print("\nStart Evaluate: ", file)
        pre = normalization(load(os.path.join(dataset_path,file,'Pre_contrast.nii.gz'))).astype(np.float32)
        pos = normalization(load(os.path.join(dataset_path,file,'Pos_contrast.nii.gz'))).astype(np.float32)
        sub = normalization(pos - pre)
        print(sub.shape)
        gt = load(os.path.join(dataset_path,file,'GT.nii.gz')).astype(np.int16)
        pre_patch, pos_patch, sub_patch, gt_patch = random_crop_3d(pre,pos,sub,gt,size)

        yield package_torch(pre_patch,pos_patch,sub_patch,gt_patch), file

def Test_all_Datasets(dataset_path,size, test_folder = 1):
    f = open(os.path.join(dataset_path,'test.txt'))
    data_list = f.read().splitlines()
    print("The number of test samples is: ", len(data_list))
    for file in data_list:
        #file = str(int(file))
        file = file.split("'")[1]
        source = file.split("_")[0]
        print("\nStart Evaluate: ", file)
        x0,_,_ = normalization(load(os.path.join(dataset_path,source,file,'P0.nii.gz')))
        x0 = x0.astype(np.float32)
        t0 = 1
        x1,x_min,x_max = normalization(load(os.path.join(dataset_path,source, file, 'P1.nii.gz')))
        x1 = x1.astype(np.float32)
        t1 = 1

        if os.path.exists(os.path.join(dataset_path,source,file,'P2.nii.gz')):
            x2 = normalization_fix(load(os.path.join(dataset_path,source, file, 'P2.nii.gz')),x_min,x_max).astype(np.float32)
            t2 = 1
        else:
            x2 = x1
            t2 = 0

        if os.path.exists(os.path.join(dataset_path,source,file,'P3.nii.gz')):
            x3 = normalization_fix(load(os.path.join(dataset_path,source, file, 'P3.nii.gz')),x_min,x_max).astype(np.float32)
            t3 = 1
        else:
            x3 = x1
            t3 = 0
        if os.path.exists(os.path.join(dataset_path,source,file,'P4.nii.gz')):
            x4 = normalization_fix(load(os.path.join(dataset_path,source, file, 'P4.nii.gz')),x_min,x_max).astype(np.float32)
            t4 = 1
        else:
            x4 = x1
            t4 = 0

        if os.path.exists(os.path.join(dataset_path,source,file,'P5.nii.gz')):
            x5 = normalization_fix(load(os.path.join(dataset_path,source, file, 'P5.nii.gz')),x_min,x_max).astype(np.float32)
            t5 = 1
        else:
            x5 = x1
            t5 = 0

        gt = load(os.path.join(dataset_path,source+'_gt',file+'.nii.gz')).astype(np.float32)
        itkimage = sitk.ReadImage(os.path.join(dataset_path,source, file, 'P1.nii.gz'))
        original_shape = gt.shape
        new_shape = cal_newshape(gt,size)
        TS_all = np.array([t0, t1, t2, t3, t4, t5]).astype(np.float32)
        Breast_mask = load(os.path.join(dataset_path,source+'_breastmask',file+'.nii.gz'))
        #pre_patch, pos_patch, sub_patch, gt_patch = random_crop_3d(pre,pos,sub,gt,size)

        yield Img_DataSet(x0,x1,x2,x3,x4,x5,gt,size),TS_all,original_shape, new_shape,itkimage, file,Breast_mask,source
            