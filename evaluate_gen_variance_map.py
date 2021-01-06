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
from model.model import SharedEncoder_var, PrivateEncoder, PrivateDecoder, Discriminator, DomainClassifier
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
'''
DATA_DIRECTORY = './data/Cityscapes/data'    # clean cityscapes images
DATA_LIST_PATH = './dataset/cityscapes_list/train_syn.txt'  # 498 images from cityscapes
SAVE_PATH = './data/Cityscapes/data/pseudo/train'
'''
'''
DATA_DIRECTORY = './data/Cityscapes/syn_fog_data'  # rename train_XXX folder to train
DATA_LIST_PATH = './dataset/cityscapes_list/train_syn.txt'  # 498 images from cityscapes
SAVE_PATH = './data/Cityscapes/syn_fog_data/pseudo/train_0.03'

'''
DATA_DIRECTORY = '/home/mxz/Seg-Uncertainty/data/Cityscapes/real_fog_data'  # rename folder to train
DATA_LIST_PATH = './util/loader/cityscapes_list/train_fz_clean.txt' # 248 clean images from foggyzurich
# DATA_LIST_PATH = './dataset/cityscapes_list/train_fz_fog_35.txt'   # 35 foggy images from foggyzurich
# DATA_LIST_PATH = './util/loader/cityscapes_list/city_fz_40.txt'   # 40 test images from foggyzurich
SAVE_PATH = './generated_imgs/variance_pred_imgs/zurich_clean/'


WEIGHT_DIR = './results/city2fz_clean_new/weight_best'   # model path

# SET = 'fz_test_40'
SET = 'fz_clean'

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

    enc_shared = SharedEncoder_var().cuda(gpu0)

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
                output1, output2 = enc_shared(inputs)
                output_batch = interp(sm(0.5 * output1 + output2))

                heatmap_batch = torch.sum(kl_distance(log_sm(output1), sm(output2)), dim=1)  # variance_map batch

                output1, output2 = enc_shared(fliplr(inputs))
                output1, output2 = fliplr(output1), fliplr(output2)
                output_batch += interp(sm(0.5 * output1 + output2))
                del output1, output2, inputs

                output1, output2 = enc_shared(inputs2)
                output_batch += interp(sm(0.5 * output1 + output2))
                output1, output2 = enc_shared(fliplr(inputs2))
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

            heatmap_tmp = heatmap_batch[i, :, :] / np.max(heatmap_batch[i, :, :])
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

num_classes = 19
CITY_DATA_PATH = '/home/mxz/Seg-Uncertainty/data/Cityscapes/real_fog_data'
DATA_LIST_PATH_TEST_IMG = './util/loader/cityscapes_list/city_fz_clean.txt'

CITY_FOG_DATA_PATH = '/home/mxz/Seg-Uncertainty/data/Cityscapes/real_fog_data'
DATA_LIST_PATH_CITY_FOG_IMG = './util/loader/cityscapes_list/train_fz_medium.txt'



DEFAULT_GPU = 1
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

parser = argparse.ArgumentParser(description='Domain Invariant Structure Extraction (DISE) \
	for unsupervised domain adaptation for semantic segmentation')

parser.add_argument('--city_data_path', type=str, default=CITY_DATA_PATH, help='the path to cityscapes.')
parser.add_argument('--target_data_path', type=str, default=CITY_FOG_DATA_PATH, help='the path to Zurich dataset.')
parser.add_argument('--data_list_path_test_img', type=str, default=DATA_LIST_PATH_TEST_IMG)
parser.add_argument('--data_list_path_gta5', type=str, default=DATA_LIST_PATH_CITY_FOG_IMG)
parser.add_argument('--gpu', type=str, default=DEFAULT_GPU)
parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)

args = parser.parse_args()

test_set = CityTestLoader(args.city_data_path, args.data_list_path_test_img, max_iters=None, crop_size=[512, 1024],
                          mean=IMG_MEAN, set='city_fz_clean')
test_loader = torch_data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

target_set = ZurichLoader(args.target_data_path, args.data_list_path_gta5, max_iters=None,
                          crop_size=[540, 960], mean=IMG_MEAN, set='fz_medium')
target_loader = torch_data.DataLoader(target_set, batch_size=1, shuffle=True, num_workers=2, pin_memory=False)

sourceloader_iter = enumerate(test_loader)
targetloader_iter = enumerate(target_loader)

upsample_1024 = nn.Upsample(size=[1024, 2048], mode='bilinear')


#enc_t.eval()
#dec_t.eval()

cty_running_metrics = runningScore(num_classes)
num_steps = test_set.__len__()
# for i_test, (images_test, name, target_data) in tqdm(enumerate(test_loader)):
for i_iter in range(num_steps):

    idx_s, source_batch = next(sourceloader_iter)
    idx_t, target_batch = next(targetloader_iter)

    images_test, name = source_batch
    target_data, target_label = target_batch

    sdatav = Variable(images_test.cuda(), volatile=True)
    tdatav = Variable(target_data.cuda(), volatile=True)

    # forwarding
    _, _, _, code_s_common = enc_shared(sdatav)
    low_t, _, _, _ = enc_shared(tdatav)

    code_t_private = enc_t(low_t)
    # transfered image
    rec_s2t = dec_t(code_s_common, code_t_private, 1)
    rec_s2t = upsample_1024(rec_s2t)
    rec_s2t = rec_s2t.detach()
    rec_s2t = rec_s2t.cpu().numpy()
    image_s2t = (rec_s2t[:, [2, 1, 0], :, :] + 1) / 2
    imgs_s = np.clip(image_s2t * 255, 0, 255).astype(np.uint8)
    imgs_s = imgs_s.squeeze()
    imgs_s = imgs_s.transpose(1, 2, 0)
    imgs_s = Image.fromarray(imgs_s)

    name = name[0][0].split('/')[-1]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    imgs_s.save(os.path.join(args.output_dir, name))

    del sdatav, tdatav, code_s_common, code_t_private, rec_s2t
    torch.cuda.empty_cache()


