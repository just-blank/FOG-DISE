import sys
import torch
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


from util.metrics import runningScore
from model.model import SharedEncoder, PrivateEncoder, PrivateDecoder, Discriminator, DomainClassifier
from util.utils import poly_lr_scheduler, adjust_learning_rate, save_models, load_models

from util.loader.CityTestLoader import CityTestLoader
from util.loader.ZurichLoader import ZurichLoader
from util.loader.cityscapes_dataset import cityscapesDataSet

import argparse

import numpy as np

from packaging import version
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
import os
from PIL import Image
from util.loader.tool import fliplr
import matplotlib.pyplot as plt
import torch.nn as nn
import yaml

torch.backends.cudnn.benchmark = True

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

# DATA_DIRECTORY = '/home/mxz/Seg-Uncertainty/data/Cityscapes/data/'  # clean cityscapes images
# DATA_LIST_PATH = './util/loader/cityscapes_list/train_syn.txt'  # 498 images from cityscapes

# SAVE_PATH = './generated_imgs/variance_pred_imgs/city_clean_38'



DATA_DIRECTORY = '/home/mxz/Seg-Uncertainty/data/Cityscapes/real_fog_data'  # rename folder to train
# DATA_LIST_PATH = './util/loader/cityscapes_list/train_fz_clean.txt' # 248 clean images from foggyzurich
# DATA_LIST_PATH = './util/loader/cityscapes_list/fz_test.txt'   # 40 test images from foggyzurich
DATA_LIST_PATH = './util/loader/cityscapes_list/train_fz_medium+test.txt'
# SAVE_PATH = './generated_imgs/variance_pred_imgs/zurich_clean_36'
SAVE_PATH = './generated_imgs/variance_pred_imgs/zurich_fog_38'

WEIGHT_DIR = './results/2clean2fz_medium_new_var/s2t1weight_best'  # model path IoU 38
# WEIGHT_DIR = './results/city2fz_clean_new_var/weight_best'  # model path IoU 36

SET = 'fz_medium'    # for zurich fog
# SET = 'fz_clean'      # for zurich clean
# SET = 'train'         # for city clean

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 248
MODEL = 'DeepLab'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument('--weight_dir', type=str, default=WEIGHT_DIR)
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--batchsize", type=int, default=12,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()


def save_heatmap(output_name):
    output, name = output_name
    fig = plt.figure()
    plt.axis('off')
    heatmap = plt.imshow(output, cmap='viridis')
    fig.colorbar(heatmap)
    fig.savefig('%s_heatmap.png' % (name.split('.jpg')[0]))
    return


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    batchsize = args.batchsize
    gpu0 = args.gpu

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model_dict = {}

    private_code_size = 8
    shared_code_channels = 2048

    enc_shared = SharedEncoder().cuda(gpu0)

    model_dict['enc_shared'] = enc_shared

    load_models(model_dict, args.weight_dir)

    enc_shared.eval()                           # load model done

    scale = 1.25
    '''
    #for cityscape 
    testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(512, 1024), resize_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                   batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)
    testloader2 = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(round(512*scale), round(1024*scale) ), resize_size=( round(1024*scale), round(512*scale)), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                   batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)
    '''

    # for foggyzurich
    testloader = data.DataLoader(
        cityscapesDataSet(args.data_dir, args.data_list, crop_size=(540, 960), resize_size=(960, 540), mean=IMG_MEAN,
                          scale=False, mirror=False, set=args.set), batch_size=batchsize, shuffle=False,
        pin_memory=True, num_workers=4)
    testloader2 = data.DataLoader(
        cityscapesDataSet(args.data_dir, args.data_list, crop_size=(round(540 * scale), round(960 * scale)),
                          resize_size=(round(960 * scale), round(540 * scale)), mean=IMG_MEAN, scale=False,
                          mirror=False, set=args.set),
        batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)


    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        # interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)  # for cityscapes
        interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)  # for foggyzurich
        # interp = nn.Upsample(size=(256, 512), mode='bilinear', align_corners=True)      # for Yan's data
    else:
        interp = nn.Upsample(size=(1024, 2048), mode='bilinear')

    sm = torch.nn.Softmax(dim=1)
    log_sm = torch.nn.LogSoftmax(dim=1)
    kl_distance = nn.KLDivLoss(reduction='none')

    for index, img_data in enumerate(zip(testloader, testloader2)):
        batch, batch2 = img_data
        image,  _, name = batch
        image2, _, name2 = batch2
        print(image.shape)

        inputs = image.cuda()
        inputs2 = image2.cuda()
        print('\r>>>>Extracting feature...%04d/%04d' % (index * batchsize, NUM_STEPS), end='')
        if args.model == 'DeepLab':
            with torch.no_grad():
                _, output1, output2,_ = enc_shared(inputs)
                output_batch = interp(sm(0.5 * output1 + output2))

                heatmap_batch = torch.sum(kl_distance(log_sm(output1), sm(output2)), dim=1)  # variance_map batch

                _, output1, output2, _ = enc_shared(fliplr(inputs))
                output1, output2 = fliplr(output1), fliplr(output2)
                output_batch += interp(sm(0.5 * output1 + output2))
                del output1, output2, inputs

                _, output1, output2, _ = enc_shared(inputs2)
                output_batch += interp(sm(0.5 * output1 + output2))
                _, output1, output2, _ = enc_shared(fliplr(inputs2))
                output1, output2 = fliplr(output1), fliplr(output2)
                output_batch += interp(sm(0.5 * output1 + output2))
                del output1, output2, inputs2
                output_batch = output_batch.cpu().data.numpy()
                heatmap_batch = heatmap_batch.cpu().data.numpy()
        elif args.model == 'DeeplabVGG' or args.model == 'Oracle':
            output_batch = enc_shared(Variable(image).cuda())
            output_batch = interp(output_batch).cpu().data.numpy()

        # output_batch = output_batch.transpose(0,2,3,1)
        # output_batch = np.asarray(np.argmax(output_batch, axis=3), dtype=np.uint8)
        output_batch = output_batch.transpose(0, 2, 3, 1)
        score_batch = np.max(output_batch, axis=3)
        output_batch = np.asarray(np.argmax(output_batch, axis=3), dtype=np.uint8)
        # output_batch[score_batch<3.2] = 255  #3.2 = 4*0.8
        for i in range(output_batch.shape[0]):
            output = output_batch[i, :, :]
            output_col = colorize_mask(output)
            output = Image.fromarray(output)
            '''
            #for cityscapes type 
            name_tmp = name[i].split('/')[-1]
            dir_name = name[i].split('/')[-2]
            save_path = args.save + '/' + dir_name
            '''
            # for foggyzurich and others
            name_tmp = name[i]
            save_path = args.save

            # save_path = re.replace(save_path, 'leftImg8bit', 'pseudo')
            # print(save_path)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            output.save('%s/%s' % (save_path, name_tmp))
            print('%s/%s' % (save_path, name_tmp))
            output_col.save('%s/%s_color.png' % (save_path, name_tmp.split('.png')[0]))

            heatmap_tmp = heatmap_batch[i, :, :] / np.max(heatmap_batch[i, :, :])   # max normalization
            fig = plt.figure()
            plt.axis('off')
            heatmap = plt.imshow(heatmap_tmp, cmap='viridis')
            # fig.colorbar(heatmap)
            fig.savefig('%s/%s_var_map.png' % (save_path, name_tmp.split('.png')[0]))

    return args.save


if __name__ == '__main__':
    with torch.no_grad():
        save_path = main()
    os.system('python compute_iou.py ./data/Cityscapes/real_fog_data/gtFine/fz_test_40 %s' % save_path)


