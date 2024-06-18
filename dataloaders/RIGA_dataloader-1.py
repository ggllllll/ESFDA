from torch.utils import data
import numpy as np
from PIL import Image, ImageFilter
from batchgenerators.utilities.file_and_folder_operations import *

def calculateCompact(image):
    #image = Image.open(image).convert('L')
    image = np.asarray(image, np.float32)
    image_od = np.zeros(image.shape, dtype=np.float32)
    image_oc = np.zeros(image.shape, dtype=np.float32)

    image_od[image == 255] = 255
    image_od[image == 128] = 255
    image_oc[image == 128] = 255

    image_od = Image.fromarray(np.uint8(image_od))
    image_oc = Image.fromarray(np.uint8(image_oc))
    edge_od = image_od.filter(ImageFilter.FIND_EDGES)
    edge_oc = image_oc.filter(ImageFilter.FIND_EDGES)

    edge_od = np.asarray(edge_od, np.float32)
    edge_oc = np.asarray(edge_oc, np.float32)

    image_od = np.asarray(image_od, np.float32)
    image_oc = np.asarray(image_oc, np.float32)

    image_od = image_od / 255
    image_oc = image_oc / 255

    edge_od = edge_od / 255
    edge_oc = edge_oc / 255

    S_od = np.sum(image_od)
    S_oc = np.sum(image_oc)

    C_od = np.sum(edge_od)
    C_oc = np.sum(edge_oc)
    #print(C, S, edge[0])
    weight_od = np.nan_to_num(np.asarray(100 * S_od / C_od ** 2, np.float32))
    weight_oc = np.nan_to_num(np.asarray(100 * S_oc / C_oc ** 2, np.float32))
    return [weight_od, weight_oc]

class RIGA_labeled_set(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=(512, 512), img_normalize=True):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.target_size = target_size
        self.img_normalize = img_normalize

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_file = join(self.root, self.img_list[item])
        label_file = join(self.root, self.label_list[item])
        img = Image.open(img_file)
        label = Image.open(label_file)
        img = img.resize(self.target_size)
        label = label.resize(self.target_size, resample=Image.NEAREST)
        weight = calculateCompact(label)
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)
        if self.img_normalize:
            for i in range(img_npy.shape[0]):
                img_npy[i] = (img_npy[i] - img_npy[i].mean()) / img_npy[i].std()
        label_npy = np.array(label)
        mask = np.zeros_like(label_npy)
        mask[label_npy > 0] = 1
        mask[label_npy == 128] = 2
        return img_npy, mask[np.newaxis], img_file, weight


class RIGA_unlabeled_set(data.Dataset):
    def __init__(self, root, img_list, target_size=(512, 512), img_normalize=True):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.len = len(img_list)
        self.target_size = target_size
        self.img_normalize = img_normalize

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_file = join(self.root, self.img_list[item])
        img = Image.open(img_file)
        img = img.resize(self.target_size)
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)
        if self.img_normalize:
            for i in range(img_npy.shape[0]):
                img_npy[i] = (img_npy[i] - img_npy[i].mean()) / img_npy[i].std()
        return img_npy, None, img_file
