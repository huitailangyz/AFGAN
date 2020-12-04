from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image

from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2, build_attn_images, build_super_images_attr
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER, ATTR_ENCODER, CNN_ENCODER_AlexNet, Self_Attention_ENCODER

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss
import os
import time
import numpy as np
import sys
import tensorboardX
import pickle

# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword, cfg):
        self.cfg = cfg
        self.output_dir = output_dir
        if self.cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        torch.cuda.set_device(self.cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = self.cfg.TRAIN.BATCH_SIZE
        self.max_epoch = self.cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = self.cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def adjust_learning_rate(self, optimizer, epoch, initial_lr, writer=None):
        """Sets the learning rate to the initial LR decayed by 0.98 each epoch"""
        lr = initial_lr * (0.98 ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if writer:
            writer.add_scalar("lr_G", lr, epoch + 1)


    def build_models(self):
        # ###################encoders######################################## #
        if self.cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        # image_encoder = CNN_ENCODER(self.cfg, self.cfg.TEXT.EMBEDDING_DIM)
        image_encoder = CNN_ENCODER_AlexNet(self.cfg, self.cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = self.cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict, strict=False)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        # text_encoder = ATTR_ENCODER(self.n_words, nhidden=self.cfg.TEXT.EMBEDDING_DIM)
        # text_encoder = RNN_ENCODER(self.cfg, self.n_words, nhidden=self.cfg.TEXT.EMBEDDING_DIM)
        text_encoder = Self_Attention_ENCODER(self.n_words, nhidden=self.cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(self.cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', self.cfg.TRAIN.NET_E)
        text_encoder.eval()

        # #######################generator and discriminators############## #
        netsD = []
        if self.cfg.GAN.B_DCGAN:
            if self.cfg.TREE.BRANCH_NUM ==1:
                from model import D_NET64 as D_NET
            elif self.cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # self.cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            # TODO: elif self.cfg.TREE.BRANCH_NUM > 3:
            netG = G_DCGAN(self.cfg)
            netsD = [nn.utils.spectral_norm(D_NET(b_jcu=False))]
        else:
            from model import D_NET64, D_NET128, D_NET256, D_PATCHGAN
            netG = G_NET(self.cfg)
            if self.cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64(self.cfg))
            if self.cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128(self.cfg))
            if self.cfg.TREE.BRANCH_NUM > 2:
                # netsD.append(D_NET256(self.cfg))
                netsD.append(D_PATCHGAN(self.cfg))
            # TODO: if self.cfg.TREE.BRANCH_NUM > 3:
        netG.apply(weights_init)
        # print(netG)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        #
        epoch = 0
        if self.cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(self.cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', self.cfg.TRAIN.NET_G)
            istart = self.cfg.TRAIN.NET_G.rfind('_') + 1
            iend = self.cfg.TRAIN.NET_G.rfind('.')
            epoch = self.cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if self.cfg.TRAIN.B_NET_D:
                Gname = self.cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = \
                        torch.load(Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)
        # ########################################################### #
        if self.cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
        return [text_encoder, image_encoder, netG, netsD, epoch]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=self.cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=self.cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if self.cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(),
                '%s/netD%d.pth' % (self.model_dir, i))
        print('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, name='current'):
        # Save images
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set = build_super_images_attr(img, captions, self.ixtoword,
                                    attn_maps, att_sze, self.cfg.TEXT.WORDS_NUM, lr_imgs=lr_img)
            # img_set, _ = \
            #     build_super_images(img, captions, cap_lens, self.ixtoword,
            #                        attn_maps, att_sze, self.cfg.TRAIN.BATCH_SIZE, self.cfg.TEXT.WORDS_NUM, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        i = -1
        img = fake_imgs[i].detach()
        region_features, _, _, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, attn_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size, self.cfg)
        img_set = build_super_images_attr(img.cpu(), captions, self.ixtoword,
                                          attn_maps, att_sze, self.cfg.TEXT.WORDS_NUM)
        # img_set, _ = \
        #     build_super_images(fake_imgs[i].detach().cpu(), captions, cap_lens,
        #                        self.ixtoword, att_maps, att_sze, self.cfg.TRAIN.BATCH_SIZE, self.cfg.TEXT.WORDS_NUM)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

    def train(self):
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = self.cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if self.cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        # gen_iterations = 0
        PRINT_INTERVAL = 200
        writer = tensorboardX.SummaryWriter(os.path.join(self.output_dir, "Log"))
        gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            data_iter = iter(self.data_loader)
            step = 0
            self.adjust_learning_rate(optimizerG, epoch, self.cfg.TRAIN.GENERATOR_LR, writer)
            for i in range(len(netsD)):
                self.adjust_learning_rate(optimizersD[i], epoch, self.cfg.TRAIN.DISCRIMINATOR_LR)
            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)
                for i in range(self.cfg.GAN.G_STEP):
                    step += 1
                    gen_iterations += 1
                    ######################################################
                    # (1) Prepare training data and Compute text embeddings
                    ######################################################
                    data = data_iter.next()
                    imgs, captions, cap_lens, class_ids, keys = prepare_data(data, self.cfg)
                    # hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, None)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)

                    #######################################################
                    # (4) Update G network: maximize log(D(G(z)))
                    ######################################################
                    # compute total loss for training G


                    # do not need to compute gradient for Ds
                    # self.set_requires_grad_value(netsD, False)
                    netG.zero_grad()
                    errG_total, G_logs = \
                        generator_loss(netsD, image_encoder, imgs[-1], fake_imgs, real_labels, words_embs, sent_emb,
                                       match_labels, cap_lens, class_ids, writer, gen_iterations, self.cfg)
                    # TODO: Delete the CA module
                    kl_loss = KL_loss(mu, logvar)
                    errG_total += kl_loss
                    G_logs += ' kl_loss: %.2f ' % kl_loss.data.cpu().numpy()
                    writer.add_scalar("Gen/kl", kl_loss.data, gen_iterations)
                    writer.add_scalar("Gen/total", errG_total.data, gen_iterations)
                    # backward and update parameters

                    errG_total.backward()
                    optimizerG.step()
                    for p, avg_p in zip(netG.parameters(), avg_param_G):
                        avg_p.mul_(0.999).add_(0.001, p.data)


                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = '|'
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                              sent_emb, real_labels, fake_labels, writer, i, gen_iterations)
                    # backward and update parameters
                    self.adjust_learning_rate(optimizersD[i], epoch, self.cfg.TRAIN.DISCRIMINATOR_LR)
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += ' errD%d: %.2f |' % (i, errD.data.cpu().numpy())

                writer.add_scalar("Dis_total", errD_total.data, gen_iterations)
                if step % PRINT_INTERVAL == 0:
                    end_t = time.time()
                    print(("| epoch %d | %d/%d | %.2f s/batch " + D_logs + G_logs) \
                          % (epoch, step, self.num_batches, (end_t - start_t) / PRINT_INTERVAL))
                    start_t = time.time()
            # save images
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            self.save_img_results(netG, fixed_noise, sent_emb,
                                  words_embs, mask, image_encoder,
                                  captions, cap_lens, epoch, name='average')
            load_params(netG, backup_para)
            #
            # self.save_img_results(netG, fixed_noise, sent_emb,
            #                       words_embs, mask, image_encoder,
            #                       captions, cap_lens,
            #                       epoch, name='current')


            # print('''[%d/%d][%d]
            #       Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
            #       % (epoch, self.max_epoch, self.num_batches,
            #          errD_total.data.cpu().numpy(), errG_total.data.cpu().numpy(),
            #          end_t - start_t))

            if epoch % self.cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or epoch == self.cfg.TRAIN.MAX_EPOCH:
                self.save_model(netG, avg_param_G, netsD, epoch)


    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling_to_pickle(self, split_dir):
        if self.cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if self.cfg.GAN.B_DCGAN:
                netG = G_DCGAN(self.cfg)
            else:
                netG = G_NET(self.cfg)
            netG.apply(weights_init)
            if self.cfg.CUDA:
                netG.cuda()
            netG.eval()
            #
            # text_encoder = RNN_ENCODER(self.cfg, self.n_words, nhidden=self.cfg.TEXT.EMBEDDING_DIM)
            text_encoder = Self_Attention_ENCODER(self.n_words, nhidden=self.cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(self.cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', self.cfg.TRAIN.NET_E)
            if self.cfg.CUDA:
                text_encoder = text_encoder.cuda()
            text_encoder.eval()

            batch_size = self.batch_size
            nz = self.cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            if self.cfg.CUDA:
                noise = noise.cuda()

            model_dir = self.cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(self.cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s_%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)


            cnt = 0
            sample_num = 5000
            raw_img_list = []
            gen_img_list = []
            label_list = []
            for step, data in enumerate(self.data_loader, 0):
                cnt += batch_size
                print('Sample: %d / %d' % (cnt, sample_num))
                if cnt >= sample_num:
                    break

                imgs, captions, cap_lens, class_ids, keys = prepare_data(data, self.cfg)

                # hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, None)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, attn_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                captions = captions.cpu().numpy()
                for j in range(batch_size):
                    raw_img = imgs[-1][j].add(1).div(2)
                    gen_img = fake_imgs[-1][j]
                    max_ = torch.max(gen_img)
                    min_ = torch.min(gen_img)
                    gen_img = (gen_img - min_) / (max_ - min_)

                    raw_img_list.append(np.uint8(raw_img.cpu().numpy() * 255))
                    gen_img_list.append(np.uint8(gen_img.detach().cpu().numpy() * 255))
                    label_list.append(captions[j])

            # img [0, 1]
            raw_img_list = np.array(raw_img_list)
            gen_img_list = np.array(gen_img_list)
            label_list = np.array(label_list)
            with open("%s/img_256_%d.pickle" % (save_dir, sample_num), 'wb') as f:
                pickle.dump([raw_img_list, gen_img_list, label_list], f, protocol=2)

    def sampling_interval(self, split_dir):
        if self.cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if self.cfg.GAN.B_DCGAN:
                netG = G_DCGAN(self.cfg)
            else:
                netG = G_NET(self.cfg)
            netG.apply(weights_init)
            if self.cfg.CUDA:
                netG.cuda()
            netG.eval()
            #
            # text_encoder = RNN_ENCODER(self.cfg, self.n_words, nhidden=self.cfg.TEXT.EMBEDDING_DIM)
            text_encoder = Self_Attention_ENCODER(self.n_words, nhidden=self.cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(self.cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', self.cfg.TRAIN.NET_E)
            if self.cfg.CUDA:
                text_encoder = text_encoder.cuda()
            text_encoder.eval()

            batch_size = self.batch_size
            nz = self.cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            if self.cfg.CUDA:
                noise = noise.cuda()

            model_dir = self.cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(self.cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s_%s/inverval_random' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0

            for step, data in enumerate(self.data_loader, 0):
                cnt += batch_size
                print('Step: %d / %d' % (step, self.num_batches))
                if step >= 2:
                    break

                imgs, captions, cap_lens, class_ids, keys = prepare_data(data, self.cfg)

                # hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, None)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                for i in range(10):
                    noise.data.normal_(0, 1)
                    fake_imgs, attn_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                    imsize = fake_imgs[-2][0].size(1)
                    fake_imgs = [nn.Upsample(size=(imsize, imsize), mode='bilinear')(x) for x in fake_imgs]


                    for j in range(batch_size):
                        raw_img = imgs[-1][j].add(1).div(2).mul(255)
                        gen_img = fake_imgs[-1][j].add(1).div(2).mul(255)
                        # max_ = torch.max(gen_img)
                        # min_ = torch.min(gen_img)
                        # gen_img = (gen_img - min_) / (max_ - min_)
                        # gen_img *= 255

                        gen_img = gen_img.permute(1, 2, 0)
                        gen_img = gen_img.data.cpu().numpy()
                        gen_img = gen_img.astype(np.uint8)
                        gen_img = Image.fromarray(gen_img)
                        fullpath = '%s/%s_%d.jpg' % (save_dir, keys[j][:-4], i)
                        gen_img.save(fullpath)

                        raw_img = raw_img.permute(1, 2, 0)
                        raw_img = raw_img.data.cpu().numpy()
                        raw_img = raw_img.astype(np.uint8)
                        raw_img = Image.fromarray(raw_img)
                        fullpath = '%s/%s' % (save_dir, keys[j])
                        raw_img.save(fullpath)




    def sampling(self, split_dir):
        if self.cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if self.cfg.GAN.B_DCGAN:
                netG = G_DCGAN(self.cfg)
            else:
                netG = G_NET(self.cfg)
            netG.apply(weights_init)
            if self.cfg.CUDA:
                netG.cuda()
            netG.eval()
            #
            text_encoder = RNN_ENCODER(self.cfg, self.n_words, nhidden=self.cfg.TEXT.EMBEDDING_DIM)
            # text_encoder = Self_Attention_ENCODER(self.n_words, nhidden=self.cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(self.cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', self.cfg.TRAIN.NET_E)
            if self.cfg.CUDA:
                text_encoder = text_encoder.cuda()
            text_encoder.eval()

            batch_size = self.batch_size
            nz = self.cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            if self.cfg.CUDA:
                noise = noise.cuda()

            model_dir = self.cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(self.cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s_%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0

            for step, data in enumerate(self.data_loader, 0):
                cnt += batch_size
                print('Step: %d / %d' % (step, self.num_batches))
                # if step >= 5:
                #     break

                imgs, captions, cap_lens, class_ids, keys = prepare_data(data, self.cfg)

                # hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, None)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, attn_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                imsize = fake_imgs[-1][0].size(1)
                fake_imgs = [nn.Upsample(size=(imsize, imsize), mode='bilinear')(x) for x in fake_imgs]
                for j in range(batch_size):
                    im_set = [imgs[-1][j].add(1).div(2).mul(255)]
                    for i in range(len(fake_imgs)):
                        img = fake_imgs[i][j]
                        max_ = torch.max(img)
                        min_ = torch.min(img)
                        img = (img - min_) / (max_ - min_)
                        img *= 255
                        im_set.append(img)
                    im = torch.cat(im_set, 2)
                    im = im.permute(1, 2, 0)
                    im = im.data.cpu().numpy()
                    im = im.astype(np.uint8)
                    im = Image.fromarray(im)
                    fullpath = '%s/%s' % (save_dir, keys[j])
                    im.save(fullpath)
                    # att_sze = attn_maps[-1].size(2)
                    # im = build_super_images_attr(torch.unsqueeze(fake_imgs[-1][j], 0),
                    #                              torch.unsqueeze(captions[j], 0), self.ixtoword,
                    #                              torch.unsqueeze(attn_maps[-1][j], 0),
                    #                              att_sze, self.cfg.TEXT.WORDS_NUM,
                    #                              lr_imgs=torch.unsqueeze(imgs[-1][j], 0), nvis=1)
                    #
                    # im, _ = build_attn_images(im_set[0], im_set[-1], torch.unsqueeze(captions[j], 0),
                    #                           [cap_lens[j]], self.ixtoword, attn_maps[-1][j], self.cfg.TEXT.WORDS_NUM)
                    # im = im.astype(np.uint8)
                    # im = Image.fromarray(im)
                    # fullpath = '%s/%s_attn.jpg' % (save_dir, keys[j][:-4])
                    # im.save(fullpath)



    def gen_example(self, data_dic):
        if self.cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            text_encoder = \
                RNN_ENCODER(self.cfg, self.n_words, nhidden=self.cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(self.cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', self.cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            # the path to save generated images
            if self.cfg.GAN.B_DCGAN:
                netG = G_DCGAN(self.cfg)
            else:
                netG = G_NET(self.cfg)
            s_tmp = self.cfg.TRAIN.NET_G[:self.cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = self.cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = self.cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.cuda()
                cap_lens = cap_lens.cuda()
                for i in range(1):  # 16
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.cuda()
                    #######################################################
                    # (1) Extract text embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)
                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                    # G attention
                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)
