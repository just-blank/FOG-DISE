import os
import os.path as osp
import numpy as np
import random
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image, ImageFile
from util.loader.autoaugment import ImageNetPolicy
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True

class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, resize_size=(1024, 512), crop_size=(512, 1024), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val', autoaug=False):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.resize_size = resize_size
        self.autoaug = autoaug
        self.h = crop_size[0]
        self.w = crop_size[1]
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        # for split in ["train", "trainval", "val"]:
         
        #https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            #label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, name.replace('leftImg8bit', 'gtFine_labelIds') ))
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        #tt = time.time()
        datafiles = self.files[index]
        name = datafiles["name"]

        image= Image.open(datafiles["img"]).convert('RGB')
        # resize
        image= image.resize(self.resize_size, Image.BICUBIC)
        if self.autoaug:
            policy = ImageNetPolicy()
            image = policy(image)

        image= np.asarray(image, np.float32)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1)) / 128.0
        # x1 = random.randint(0, image.shape[1] - self.h)
        # y1 = random.randint(0, image.shape[2] - self.w)
        # image = image[:, x1:x1+self.h, y1:y1+self.w]
        #label_copy = label_copy[x1:x1+self.h, y1:y1+self.w]
            #label_copy = np.flip(label_copy, axis = 1)
        #print('Time used: {} sec'.format(time.time()-tt))
        return image.copy(), np.array(size), name


if __name__ == '__main__':
    dst = cityscapesDataSet('/home/mxz/Seg-Uncertainty/data/Cityscapes/real_fog_data', './cityscapes_list/fz_test.txt', mean=(0,0,0), set = 'fz_test_40')
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, _, _ = data
        if i == 1:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            # labels = torchvision.utils.make_grid(labels).numpy()
            # labels = np.transpose(labels, (1, 2, 0))
            # labels = labels[:, :, ::-1]
            #plt.imshow(img)
            #plt.imshow(labels)
            #plt.show()
            img = Image.fromarray(np.uint8(img) )
            img.save('./Cityscape_Demo.jpg')

