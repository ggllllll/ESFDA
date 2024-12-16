import torch.utils.data as data
import os
import glob
import torch
import numpy as np
import random,math
from .transformations import get_transform,get_transform_strong_Weak
from PIL import Image


class MyDataset(data.Dataset):
    def __init__(self, rootdir, sites, img_list, label_list, phase='train', split_train=False, weak_strong_aug=False):

        self.rootdir = rootdir
        self.sites = sites
        self.phase = phase
        self.weak_strong_aug = weak_strong_aug
        self.all_data_path = img_list
        self.all_seg_path = label_list
        self.name_list = []
        self.site_list = []

        if self.weak_strong_aug:
            self.augmenter_w, self.augmenter_s = get_transform_strong_Weak(self.phase,New_size=(512,512))
        else:
            self.augmenter = get_transform(self.phase,New_size=(512,512))
        # for site in sites:
        #     if split_train:
        #         # data_dir = os.path.join(self.rootdir,site,'train')
        #         data_dir = os.path.join(self.rootdir,'RIGA-pseudo-our',site, 'Unlabeled','generate_pseudo.csv')
        #         if site == 'Source':
        #             data_dir = []
        #     else:
        #         # data_dir = os.path.join(self.rootdir,site,'test')
        #         data_dir = os.path.join(self.rootdir,site+'.csv')
        #     img_list, label_list = convert_labeled_list(tuple([data_dir]), r=1)

            # for data_name in img_list:
            #     self.name_list.append(data_name.split('/')[-1][:-4])
            #     self.all_data_path.append(os.path.join(rootdir,data_name))
            #     self.site_list.append(site)

                
    def __getitem__(self, index):
        name = self.all_data_path[index].split('/')[-1]
        img_file = os.path.join(self.rootdir, self.all_data_path[index])
        label_file = os.path.join(self.rootdir, self.all_seg_path[index])
        img = Image.open(img_file)
        label = Image.open(label_file)
        img = img.resize((512, 512))
        label = label.resize((512, 512), resample=Image.NEAREST)

        img_npy = np.array(img).astype(np.float32)

        # for i in range(img_npy.shape[0]):
        #     img_npy[i] = (img_npy[i] - img_npy[i].mean()) / img_npy[i].std()

        label_npy = np.array(label)
        mask = np.zeros_like(label_npy)
        mask[label_npy > 0] = 1
        mask[label_npy == 128] = 2
        # return img_npy, mask[np.newaxis], img_file
        if self.weak_strong_aug:
            transformed_w = self.augmenter_w(image=img_npy, mask=mask)
            img_w = transformed_w['image']
            seg = transformed_w['mask']
            transformed_s = self.augmenter_s(image=img_w.numpy().transpose((1,2,0)))
            img_s = transformed_s['image']
            for i in range(img_w.shape[0]):
                img_w[i] = (img_w[i] - img_w[i].mean()) / img_w[i].std()

            for i in range(img_s.shape[0]):
                img_s[i] = (img_s[i] - img_s[i].mean()) / img_s[i].std()

            return img_w, img_s, seg, name
        
        else:
            transformed = self.augmenter(image=img_npy, mask=mask)
            img = transformed['image']
            img = img.to(torch.float32)
            seg = transformed['mask']
            seg = seg.to(torch.long)

            for i in range(img.shape[0]):
                img[i] = (img[i] - img[i].mean()) / img[i].std()

            return img, seg, name
    
    def __len__(self):
        return len(self.all_data_path)
    
class PatientDataset(data.Dataset):
    def __init__(self, rootdir, sites, phase='train',split_train=False, weak_strong_aug=False):

        self.rootdir = rootdir
        self.sites = sites
        self.phase = phase
        
        self.augmenter = get_transform(self.phase,New_size=(512,512))
        self.patients = []
        self.patients_slices = {}
        self.all_data_path = []
        self.name_list = []
        start,end = 0,0
        for site in sites:
            if split_train:
                data_dir = os.path.join(self.rootdir,site,'train')
            else:
                data_dir = os.path.join(self.rootdir,site,'test')
            patients = sorted(list(set([data_name.split('_')[0] for data_name in os.listdir(data_dir)])))
            patients_to_idxs = {patient:idx+len(self.patients) for idx,patient in enumerate(patients)}
            idxs = list(patients_to_idxs.values())
            patients_slice_num = {i:0 for i in idxs}
            patients_slices = {patient:[] for patient in patients}
            for patient in patients:
                idx = patients_to_idxs[patient]
                slice_list = glob.glob(os.path.join(data_dir,patient + '*'))
                for slice_path in slice_list:
                    patients_slice_num[idx] += 1
                    self.all_data_path.append(slice_path)
                    self.name_list.append(os.path.basename(slice_path)[:-4])
                end = len(self.all_data_path)
                patients_slices[patient] = [start,end]
                start = end
            self.patients.extend(patients)
            self.patients_slices.update(patients_slices)        
                
    def __getitem__(self, index):
        
        data = np.load(self.all_data_path[index])
        name = self.name_list[index]
        img = data['image'].astype(np.float32)
        seg = data['label']
        if self.weak_strong_aug:
            transformed_w = self.augmenter_w(image=img, mask=seg)
            img_w = transformed_w['image']
            seg = transformed_w['mask']
            transformed_s = self.augmenter_s(image=img_w.numpy().transpose((1,2,0)))
            img_s = transformed_s['image']
            return img_w, img_s, seg, name
        
        else:
            transformed = self.augmenter(image=img, mask=seg)
            img = transformed['image']
            img = img.to(torch.float32)
            seg = transformed['mask']
            seg = seg.to(torch.long)
            return img, seg, name

    def __len__(self):
        return len(self.all_data_path)
    
class MyBatchSampler(data.Sampler):
    
    def __init__(self, data_source, batch_size, random=True):
        super(MyBatchSampler, self).__init__(data_source)
        self.data_source = data_source

        self.batch_size = batch_size
        self.random = random

        patients = data_source.patients
        # random.shuffle(patients)
        # patients = ['0036','0024','0002','0035','0010','0040']
        self.patients = patients

    def __iter__(self):
        print('======= start __iter__ =======')
        
        if self.random:
            i = 0
            while i < len(self):
                for patient in self.patients:
                    start, end = self.data_source.patients_slices[patient]
                    batch = random.sample(range(start,end),self.batch_size)

                    assert len(batch) == self.batch_size
                    # random.shuffle(batch)
                    yield batch
                    batch = []
                    i += 1
        else:
            for patient in self.patients:
                start, end = self.data_source.patients_slices[patient]
                k = start
                while k+self.batch_size<end:
                    batch = list(range(k,k+self.batch_size))
                    yield batch
                    k = k+self.batch_size
                if k+self.batch_size>=end:
                    yield list(range(k,end))

    def __len__(self):
        if self.random:
            return len(self.patients)
        else:
            length = 0
            for patient in self.patients:
                start, end = self.data_source.patients_slices[patient]
                length += math.ceil((end-start)/self.batch_size)
            return int(length)


if __name__ == '__main__':
    data_root = '/mnt/sda/qinji/Domain_Adaptation/data/Abdomen_Data/'
    sites = ['CT']
    # dataset = PatientDataset(data_root,sites,'abdomen','test',True,16)
    # patient_sampler = MyBatchSampler(dataset,16,False)
    # dataloader = data.DataLoader(dataset,batch_size=1,batch_sampler=patient_sampler,num_workers=8)
    dataset = MyDataset(data_root,sites,'abdomen','train',True,weak_strong_aug=True)
    dataloader = data.DataLoader(dataset,batch_size=8,num_workers=4)
    # for images,segs,names in dataloader:
    #     print(images.shape,segs.shape,names)
    #     break
    for image_w,image_s,segs,names in dataloader:
        print(image_w.shape,image_s.shape,segs.shape,names)
        break