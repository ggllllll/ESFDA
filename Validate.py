import argparse
import os
import tqdm
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloaders.transform import collate_fn_tr, collate_fn_ts
from dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from dataloaders.convert_csv_to_list import convert_labeled_list

from PIL import Image
from sklearn.cluster import KMeans
from networks.unet import UNet
from plot.evaluate_mu import eval_print_all_CU
from utils.util import sigmoid_entropy
import torch.cuda

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str,
                        default=r'H:\ggl\log\DS\model.model')
    parser.add_argument('--batchsize', type=int, default=1)  # 1
    parser.add_argument('-g', '--gpu', type=int, default=3)
    parser.add_argument('--root', default=r'E:\GGL\ggl\RIGAPlus')
    parser.add_argument('--tr_csv', help='training csv file.', default=[r'E:\GGL\ggl\RIGAPlus\DS.csv'])
    args = parser.parse_args()
    inference_tag = 'DS'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file
    visualization_folder = r'E:\GGL\ggl\SFDA-our\5'
    os.makedirs(visualization_folder, exist_ok=True)
    tr_csv = tuple(args.tr_csv)
    root_folder = args.root
    tr_img_list, tr_label_list = convert_labeled_list(tr_csv, r=1)
    tr_dataset = RIGA_unlabeled_set(root_folder, tr_img_list)
    train_loader = torch.utils.data.DataLoader(tr_dataset,
                                               batch_size=args.batchsize,
                                               num_workers=1,
                                               shuffle=False,
                                               pin_memory=True,
                                               collate_fn=collate_fn_tr)

    model = UNet()
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' % (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.eval()
    pseudo_label_name = []


    with torch.no_grad():
        for batch_idx, (sample) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), ncols=80, leave=False):
            data, img_name = sample['data'], sample['name']
            data = torch.from_numpy(data).to(dtype=torch.float32)
            if torch.cuda.is_available():
                data = data.cuda()
            data = Variable(data)
            output = model(data)
            output_sigmoid = torch.sigmoid(output).cpu()
            for i in range(data.shape[0]):
                case_seg = np.zeros((512, 512))
                case_seg[output_sigmoid[i][0] > 0.5] = 255
                case_seg[output_sigmoid[i][1] > 0.5] = 128
                case_seg_f = Image.fromarray(case_seg.astype(np.uint8)).resize((512, 512), resample=Image.NEAREST)
                case_seg_f.save(
                    os.path.join(visualization_folder, img_name[i].split('/')[-1].replace('.tif', '-1.tif')))

    eval_print_all_CU(visualization_folder + '/', os.path.join(root_folder, 'RIGA-mask', inference_tag, 'Labeled'))
    with open(os.path.join(visualization_folder, 'generate_pseudo.csv'), 'w') as f:
        f.write('image,mask\n')
        for i in pseudo_label_name:
            f.write('RIGA/{}/Unlabeled/{},RIGA-pseudo-our/{}/Unlabeled/{}\n'.format(inference_tag, i, inference_tag, i))
