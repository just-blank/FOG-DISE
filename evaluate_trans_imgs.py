import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import torchvision.models as models
import torch.utils.data as torch_data
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
import os

# from tensorboardX import SummaryWriter
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm

from util.metrics import runningScore
from model.model import SharedEncoder, PrivateEncoder, PrivateDecoder, Discriminator, DomainClassifier
from util.utils import poly_lr_scheduler, adjust_learning_rate, save_models, load_models

from util.loader.CityTestLoader import CityTestLoader
from util.loader.ZurichLoader import ZurichLoader

num_classes = 19
CITY_DATA_PATH = '/home/mxz/Seg-Uncertainty/data/Cityscapes/real_fog_data'
DATA_LIST_PATH_TEST_IMG = './util/loader/cityscapes_list/city_fz_clean.txt'

CITY_FOG_DATA_PATH = '/home/mxz/Seg-Uncertainty/data/Cityscapes/real_fog_data'
DATA_LIST_PATH_CITY_FOG_IMG = './util/loader/cityscapes_list/train_fz_medium.txt'

WEIGHT_DIR = './results/fz_clean/weight_60000'
OUTPUT_DIR = './generated_imgs/2clean2fz_medium/transfered_imgs/'
DEFAULT_GPU = 1
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

parser = argparse.ArgumentParser(description='Domain Invariant Structure Extraction (DISE) \
	for unsupervised domain adaptation for semantic segmentation')
parser.add_argument('--weight_dir', type=str, default=WEIGHT_DIR)
parser.add_argument('--city_data_path', type=str, default=CITY_DATA_PATH, help='the path to cityscapes.')
parser.add_argument('--target_data_path', type=str, default=CITY_FOG_DATA_PATH, help='the path to Zurich dataset.')
parser.add_argument('--data_list_path_test_img', type=str, default=DATA_LIST_PATH_TEST_IMG)
parser.add_argument('--data_list_path_gta5', type=str, default=DATA_LIST_PATH_CITY_FOG_IMG)
parser.add_argument('--gpu', type=str, default=DEFAULT_GPU)
parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)

args = parser.parse_args()

test_set   = CityTestLoader(args.city_data_path, args.data_list_path_test_img, max_iters=None, crop_size=[512, 1024], mean=IMG_MEAN, set='city_fz_clean')
test_loader= torch_data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

target_set = ZurichLoader(args.target_data_path, args.data_list_path_gta5, max_iters=None,
                        crop_size=[540, 960], mean=IMG_MEAN, set='fz_medium')
target_loader = torch_data.DataLoader(target_set, batch_size=1, shuffle=True, num_workers=2, pin_memory=False)

sourceloader_iter = enumerate(test_loader)
targetloader_iter = enumerate(target_loader)

upsample_1024 = nn.Upsample(size=[1024, 2048], mode='bilinear')

model_dict = {}

private_code_size = 8
shared_code_channels = 2048

enc_shared = SharedEncoder().cuda()
#dclf1      = DomainClassifier().cuda()
#dclf2      = DomainClassifier().cuda()
#enc_s      = PrivateEncoder(64, private_code_size).cuda()
enc_t      = PrivateEncoder(64, private_code_size).cuda()
#dec_s      = PrivateDecoder(shared_code_channels, private_code_size).cuda()
dec_t      = PrivateDecoder(shared_code_channels, private_code_size).cuda()
#dis_s2t    = Discriminator().cuda()
#dis_t2s    = Discriminator().cuda()

model_dict['enc_shared'] = enc_shared

model_dict['enc_t'] = enc_t
#model_dict['dec_s'] = dec_s
model_dict['dec_t'] = dec_t


load_models(model_dict, args.weight_dir)

enc_shared.eval()
enc_t.eval()
dec_t.eval()

cty_running_metrics = runningScore(num_classes)    
num_steps = test_set.__len__()
#for i_test, (images_test, name, target_data) in tqdm(enumerate(test_loader)):
for i_iter in range(num_steps):

    idx_s, source_batch = next(sourceloader_iter)
    idx_t, target_batch = next(targetloader_iter)

    images_test, name = source_batch
    target_data, target_label = target_batch


    sdatav = Variable(images_test.cuda(), volatile=True)
    tdatav = Variable(target_data.cuda(),volatile=True)
    
    # forwarding
    _, _, _, code_s_common = enc_shared(sdatav)
    low_t, _, _, _ = enc_shared(tdatav)

    code_t_private    = enc_t(low_t)
    # transfered image
    rec_s2t = dec_t(code_s_common, code_t_private, 1)
    rec_s2t = upsample_1024(rec_s2t)
    rec_s2t = rec_s2t.detach()
    rec_s2t = rec_s2t.cpu().numpy()
    image_s2t = (rec_s2t[:,[2, 1, 0],:,:]+1)/2
    imgs_s = np.clip(image_s2t*255,0,255).astype(np.uint8)
    imgs_s = imgs_s.squeeze()
    imgs_s = imgs_s.transpose(1,2,0)
    imgs_s = Image.fromarray(imgs_s)

    name = name[0][0].split('/')[-1]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    imgs_s.save(os.path.join(args.output_dir, name))

    del sdatav,tdatav,code_s_common,code_t_private,rec_s2t
    torch.cuda.empty_cache()


    '''
    _, _, pred, _ = enc_shared(images_test)
    pred = upsample_1024(pred)

    pred = pred.data.cpu().numpy()[0]
    pred = pred.transpose(1,2,0)
    pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8)
    pred = np.asarray(test_set.convert_back_to_id(pred), dtype=np.uint8)
    pred = Image.fromarray(pred)
    
    name = name[0][0].split('/')[-1]
    if not os.path.exists(args.output_dir):
    	os.makedirs(args.output_dir)
    pred.save(os.path.join(args.output_dir, name))
    '''
