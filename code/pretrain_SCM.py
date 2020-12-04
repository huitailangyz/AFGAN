from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images_attr
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

from model import RNN_ENCODER, CNN_ENCODER, ATTR_ENCODER, CNN_ENCODER_SPHERE, CNN_ENCODER_AlexNet, Self_Attention_ENCODER, Self_Attention_concat_ENCODER

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import tensorboardX

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


UPDATE_INTERVAL = 100
IMAGE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Train a SCM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/SCM/CelebA.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args




def sampling(dataloader, cnn_model, rnn_model, batch_size,
          labels, ixtoword, image_dir, count):
    cnn_model.eval()
    rnn_model.eval()
    for step, data in enumerate(dataloader):
        if step > count:
            exit()
        # print('step', step)

        imgs, captions, cap_lens, \
        class_ids, keys = prepare_data(data, cfg)

        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nsef
        with torch.no_grad():
            words_features, sent_code, _ = cnn_model(imgs[-1])
        # print("words_feature: ", words_features.shape, " sent_code: ", sent_code.shape)
        # --> batch_size x nef x 17*17
        nef, att_sze = words_features.size(1), words_features.size(2)
        # words_features = words_features.view(batch_size, nef, -1)

        # hidden = rnn_model.init_hidden(batch_size)
        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        with torch.no_grad():
            words_emb, sent_emb = rnn_model(captions, cap_lens, None)
        # print("words_emb: ", words_emb.shape, " sent_emb: ", sent_emb.shape)
        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                 cap_lens, class_ids, batch_size, cfg)
        img_set = build_super_images_attr(imgs[-1].cpu(), captions,
                                              ixtoword, attn_maps, att_sze, cfg.TEXT.WORDS_NUM)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/attention_maps_%d.png' % (image_dir, step)
            im.save(fullpath)


def train(dataloader, cnn_model, rnn_model, batch_size,
          labels, optimizer, epoch, ixtoword, image_dir, writer):
    cnn_model.train()
    rnn_model.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    count = epoch * len(dataloader)
    start_time = time.time()
    for step, data in enumerate(dataloader):
        count += 1
        # print('step', step)
        rnn_model.zero_grad()
        cnn_model.zero_grad()

        imgs, captions, cap_lens, \
            class_ids, keys = prepare_data(data, cfg)

        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nsef
        words_features, sent_code, _ = cnn_model(imgs[-1])
        # print("words_feature: ", words_features.shape, " sent_code: ", sent_code.shape)
        # --> batch_size x nef x 17*17
        nef, att_sze = words_features.size(1), words_features.size(2)
        # words_features = words_features.view(batch_size, nef, -1)

        # hidden = rnn_model.init_hidden(batch_size)
        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_emb, sent_emb = rnn_model(captions, cap_lens, None)
        # print("words_emb: ", words_emb.shape, " sent_emb: ", sent_emb.shape)
        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                 cap_lens, class_ids, batch_size, cfg)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size, cfg)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data
        writer.add_scalar("word_loss0", w_loss0.data, count)
        writer.add_scalar("word_loss1", w_loss1.data, count)
        writer.add_scalar("sentence_loss0", s_loss0.data, count)
        writer.add_scalar("sentence_loss1", s_loss1.data, count)
        writer.add_scalar("total_loss", loss, count)
        loss.backward()

        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(rnn_model.parameters(),
                                      cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()

        if step % UPDATE_INTERVAL == 0:
            s_cur_loss0 = s_total_loss0.data / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1.data / UPDATE_INTERVAL
            w_cur_loss0 = w_total_loss0.data / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1.data / UPDATE_INTERVAL

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | {:5.2f} ms/batch | '
                  's_loss {:5.2f} {:5.2f} | '
                  'w_loss {:5.2f} {:5.2f}'
                  .format(epoch, step, len(dataloader),
                          elapsed * 1000. / UPDATE_INTERVAL,
                          s_cur_loss0, s_cur_loss1,
                          w_cur_loss0, w_cur_loss1))
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            start_time = time.time()
        if step % IMAGE_INTERVAL == 0:
            # attention Maps
            img_set = build_super_images_attr(imgs[-1].cpu(), captions,
                                   ixtoword, attn_maps, att_sze, cfg.TEXT.WORDS_NUM)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/attention_maps_%d_%d.png' % (image_dir, epoch, step)
                im.save(fullpath)
    return count


def evaluate(dataloader, cnn_model, rnn_model, batch_size):
    cnn_model.eval()
    rnn_model.eval()
    s_total_loss = 0
    w_total_loss = 0
    for step, data in enumerate(dataloader, 0):
        real_imgs, captions, cap_lens, \
                class_ids, keys = prepare_data(data, cfg)
        with torch.no_grad():
            words_features, sent_code, _ = cnn_model(real_imgs[-1])
        # nef = words_features.size(1)
        # words_features = words_features.view(batch_size, nef, -1)

        with torch.no_grad():
            words_emb, sent_emb = rnn_model(captions, cap_lens, None)

        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
                                            cap_lens, class_ids, batch_size, cfg)
        w_total_loss += (w_loss0 + w_loss1).data

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size, cfg)
        s_total_loss += (s_loss0 + s_loss1).data

        if step == 50:
            break

    s_cur_loss = s_total_loss.data / step
    w_cur_loss = w_total_loss.data / step

    return s_cur_loss, w_cur_loss


def build_models():
    # build model ############################################################
    # text_encoder = RNN_ENCODER(cfg, dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    text_encoder = Self_Attention_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    # text_encoder = ATTR_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    # text_encoder = Self_Attention_concat_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    # image_encoder = CNN_ENCODER(cfg, cfg.TEXT.EMBEDDING_DIM)
    # image_encoder = CNN_ENCODER_SPHERE(cfg, cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER_AlexNet(cfg, cfg.TEXT.EMBEDDING_DIM)
    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)
    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    return text_encoder, image_encoder, labels, start_epoch


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)




    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Scale(imsize)
    dataset = TextDataset(cfg.DATA_DIR, cfg, 'train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)

    print("Words: %d" % dataset.n_words)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #
    dataset_val = TextDataset(cfg.DATA_DIR, cfg, 'test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, drop_last=True,
        shuffle=False, num_workers=int(cfg.WORKERS))

    # Train ##############################################################
    text_encoder, image_encoder, labels, start_epoch = build_models()
    print(text_encoder)
    print(image_encoder)
    print("Finish build model")
    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)

    for i in text_encoder.named_parameters():
        print(i[0], i[1].shape)
    for i in image_encoder.named_parameters():
        if i[1].requires_grad:
            print(i[0], i[1].shape)
    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.

    ##########################################################################
    #   Test
    ##########################################################################
    if not cfg.TRAIN.FLAG:
        timestamp = os.path.basename(os.path.dirname(os.path.dirname(cfg.TRAIN.NET_E)))
        output_dir = os.path.join('../output', timestamp)
        sample_dir = os.path.join(output_dir, 'Sample')
        mkdir_p(sample_dir)
        sampling(dataloader_val, image_encoder, text_encoder, batch_size,
                 labels, dataset.ixtoword, sample_dir, 10)
    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True


    try:
        lr = cfg.TRAIN.ENCODER_LR
        writer = tensorboardX.SummaryWriter(os.path.join(output_dir, 'Log'))
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            writer.add_scalar("lr", lr, epoch+1)
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            epoch_start_time = time.time()
            count = train(dataloader, image_encoder, text_encoder,
                          batch_size, labels, optimizer, epoch,
                          dataset.ixtoword, image_dir, writer)
            print('-' * 89)
            if len(dataloader_val) > 0:
                # if cfg.CUDA:
                #     torch.cuda.empty_cache()
                s_loss, w_loss = evaluate(dataloader_val, image_encoder,
                                          text_encoder, batch_size)
                print('| end epoch {:3d} | valid loss '
                      '{:5.2f} {:5.2f} | lr {:.5f}|'
                      .format(epoch, s_loss, w_loss, lr))
            print('-' * 89)
            if lr > cfg.TRAIN.ENCODER_LR/10.:
                lr *= 0.98

            if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                epoch == cfg.TRAIN.MAX_EPOCH):
                torch.save(image_encoder.state_dict(),
                           '%s/image_encoder%d.pth' % (model_dir, epoch))
                torch.save(text_encoder.state_dict(),
                           '%s/text_encoder%d.pth' % (model_dir, epoch))
                print('Save G/Ds models.')
    except KeyboardInterrupt:
        writer.close()
        print('-' * 89)
        print('Exiting from training early')
