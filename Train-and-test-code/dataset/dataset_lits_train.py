import random
import numpy as np
import SimpleITK as sitk
import os
from torch.utils.data import Dataset
  
class Lits_DataSet(Dataset):
    def __init__(self,root, size = (32,128,128)):
        self.root = root
        self.size = size

        f = open(os.path.join(self.root,'train.txt'))
        self.filename = f.read().splitlines()

    def __getitem__(self, index):

        file = self.filename[index]
        file = file.split("'")[1]
        source = file.split("_")[0]

        x0,_,_ = self.normalization(self.load(os.path.join(self.root,source,file,'P0.nii.gz')))
        x0 = x0.astype(np.float32)
        t0 = 1
        x1,x_min,x_max = self.normalization(self.load(os.path.join(self.root,source, file, 'P1.nii.gz')))
        x1 = x1.astype(np.float32)
        t1 = 1

        if os.path.exists(os.path.join(self.root,source,file,'P2.nii.gz')):
            x2 = self.normalization_fix(self.load(os.path.join(self.root,source, file, 'P2.nii.gz')),x_min,x_max).astype(np.float32)
            t2 = 1
        else:
            x2 = x1
            t2 = 0

        if os.path.exists(os.path.join(self.root,source,file,'P3.nii.gz')):
            x3 = self.normalization_fix(self.load(os.path.join(self.root,source, file, 'P3.nii.gz')),x_min,x_max).astype(np.float32)
            t3 = 1
        else:
            x3 = x1
            t3 = 0
        if os.path.exists(os.path.join(self.root,source,file,'P4.nii.gz')):
            x4 = self.normalization_fix(self.load(os.path.join(self.root,source, file, 'P4.nii.gz')),x_min,x_max).astype(np.float32)
            t4 = 1
        else:
            x4 = x1
            t4 = 0

        if os.path.exists(os.path.join(self.root,source,file,'P5.nii.gz')):
            x5 = self.normalization_fix(self.load(os.path.join(self.root,source, file, 'P5.nii.gz')),x_min,x_max).astype(np.float32)
            t5 = 1
        else:
            x5 = x1
            t5 = 0

        gt = self.load(os.path.join(self.root,source+'_gt',file+'.nii.gz')).astype(np.float32)

        gt_patch, cor_box = self.random_crop_3d(gt,self.size,'partial')
        x0_patch = self.crop_path(x0, cor_box)
        x1_patch = self.crop_path(x1, cor_box)
        x2_patch = self.crop_path(x2, cor_box)
        x3_patch = self.crop_path(x3, cor_box)
        x4_patch = self.crop_path(x4, cor_box)
        x5_patch = self.crop_path(x5, cor_box)
        TS_all = np.array([t0,t1,t2,t3,t4,t5]).astype(np.float32)
        TS_two = np.array([t0,t1,0,0,0,0]).astype(np.float32)
        TS_three = np.array([t0,t1,t2,0,0,0]).astype(np.float32)
        TS_four = np.array([t0, t1, t2, t3, 0, 0]).astype(np.float32)

        return x0_patch[np.newaxis,:],x1_patch[np.newaxis,:],x2_patch[np.newaxis,:],x3_patch[np.newaxis,:],x4_patch[np.newaxis,:],x5_patch[np.newaxis,:],gt_patch[np.newaxis,:],TS_all,TS_two,TS_three,TS_four

    def __len__(self):
        return len(self.filename)

    def random_crop_3d(self,gt,crop_size,pattern = 'contain'):

        cor_box = self.maskcor_extract_3d(gt)
        if pattern == 'contain':
            random_x_min, random_x_max = max(cor_box[0,1] - crop_size[0], 0), min(cor_box[0,0], gt.shape[0]-crop_size[0])
            random_y_min, random_y_max = max(cor_box[1,1] - crop_size[1], 0), min(cor_box[1,0], gt.shape[1]-crop_size[1])
            random_z_min, random_z_max = max(cor_box[2,1] - crop_size[2], 0), min(cor_box[2,0], gt.shape[2]-crop_size[2])
            if random_x_min >random_x_max:
                random_x_min, random_x_max = cor_box[0,0], cor_box[0,1] - crop_size[0]
            if random_y_min >random_y_max:
                random_y_min, random_y_max = cor_box[1,0], cor_box[1,1] - crop_size[1]
            if random_z_min > random_z_max:
                random_z_min, random_z_max = cor_box[2, 0], cor_box[2, 1] - crop_size[2]

            x_random = random.randint(random_x_min, random_x_max)
            y_random = random.randint(random_y_min, random_y_max)
            z_random = random.randint(random_z_min, random_z_max)
        elif pattern == 'partial':
            random_x_min, random_x_max = max(cor_box[0, 0] - crop_size[0], 0), min(cor_box[0, 1],gt.shape[0] - crop_size[0])
            random_y_min, random_y_max = max(cor_box[1, 0] - crop_size[1], 0), min(cor_box[1, 1],gt.shape[1] - crop_size[1])
            random_z_min, random_z_max = max(cor_box[2, 0] - crop_size[2], 0), min(cor_box[2, 1],gt.shape[2] - crop_size[2])
            x_random = random.randint(random_x_min, random_x_max)
            y_random = random.randint(random_y_min, random_y_max)
            z_random = random.randint(random_z_min, random_z_max)
        else:
            print('No sampling pattern!')


        gt_patch = gt[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]

        crop_box = np.zeros([3, 2], dtype=np.int)
        crop_box[0, 0], crop_box[0, 1] = x_random, x_random + crop_size[0]
        crop_box[1, 0], crop_box[1, 1] = y_random, y_random + crop_size[1]
        crop_box[2, 0], crop_box[2, 1] = z_random, z_random + crop_size[2]

        return gt_patch,crop_box


    def crop_path(self, img, crop_box):
        img_patch = img[crop_box[0,0]:crop_box[0,1], crop_box[1,0]:crop_box[1,1], crop_box[2,0]:crop_box[2,1]]
        return img_patch

    def normalization(self, img, lmin=1, rmax=None, dividend=None, quantile=1):
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


    def normalization_fix(self, img, minval, maxval, lmin=1):
        newimg = img.copy()
        newimg = newimg.astype(np.float32)
        if lmin is not None:
            newimg[newimg < lmin] = lmin

        newimg = (np.asarray(newimg).astype(np.float32) - minval) / (maxval - minval)
        return newimg

    def load(self,file):
        itkimage = sitk.ReadImage(file)
        image = sitk.GetArrayFromImage(itkimage)
        return image

    def maskcor_extract_3d(self,mask, padding=(0, 0, 0)):
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






