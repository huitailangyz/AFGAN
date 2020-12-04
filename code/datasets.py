# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def prepare_data(data, cfg):
    imgs, attrs, attrs_lens, class_ids, keys = data

    real_imgs = []
    for i in range(len(imgs)):
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    attrs = attrs.squeeze()
    class_ids = class_ids.numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    if cfg.CUDA:
        attrs = Variable(attrs).cuda()
        attrs_lens = Variable(attrs_lens).cuda()
    else:
        attrs = Variable(attrs)
        attrs_lens = Variable(attrs_lens)
    return [real_imgs, attrs, attrs_lens,
            class_ids, keys]


def get_imgs(cfg, img_path, imsize, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size

    if transform is not None:
        img = transform(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Scale(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir, cfg, split='train',
                 base_size=64,
                 transform=None, target_transform=None):
        self.cfg = cfg
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.filepaths, self.ids, self.attrs, \
            self.ixtoword, self.wordtoix, self.n_words = self.load_attn_data(data_dir, split)

        self.number_example = len(self.filenames)


    def load_attn_data(self, data_dir, split):
        if split == "train":
            path = os.path.join(data_dir, 'train.pickle')
        else:
            path = os.path.join(data_dir, 'test.pickle')
        with open(path, 'rb') as f:
            x = pickle.load(f)
            filenames, filepaths, ids, attrs = x[0], x[1], x[2], x[3]
            del x
        # Load the attribution dict
        path = os.path.join(data_dir, 'dict.pickle')
        with open(path, 'rb') as f:
            x = pickle.load(f)
            ixtoword, wordtoix = x[0], x[1]
            del x
            n_words = len(ixtoword)
        return filenames, filepaths, ids, attrs, ixtoword, wordtoix, n_words


    def get_attn(self, sent_ix):
        # a list of indices for a sentence
        sent_attr = np.asarray(self.attrs[sent_ix]).astype('int64')
        x_len = len(sent_attr)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.cfg.TEXT.WORDS_NUM, 1), dtype='float32')
        x[sent_attr] = 1
        return x, x_len

    def __getitem__(self, index):
        img_name = self.filepaths[index]
        key = self.filenames[index]
        cls_id = self.ids[index]

        imgs = get_imgs(self.cfg, img_name, self.imsize,
                        self.transform, normalize=self.norm)
        # random select a sentence
        attrs, attr_len = self.get_attn(index)
        return imgs, attrs, attr_len, cls_id, key


    def __len__(self):
        return len(self.filenames)
