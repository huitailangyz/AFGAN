import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import numpy as np

from GlobalAttention import func_attention


def multi_binary_cross_entropy(F_label, T_label):
    '''
    :param T_label:  Shape: [BATCH_SIZE, CLASS_NUM]
    :param F_label:  Shape: [BATCH_SIZE, CLASS_NUM]
    :return:         Shape: [1]
    '''
    eps = 10**(-10)
    delete_list = [2, 6, 7, 10, 13, 14, 16, 18, 19, 22, 25, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    attr_list = [i for i in range(40) if i not in delete_list]
    F_label = torch.sigmoid(F_label)
    F_label = F_label[:, attr_list]
    T_label = T_label[:, attr_list]
    loss = -T_label * torch.log(F_label+eps) - (1-T_label) * torch.log(1-F_label+eps)
    loss = torch.mean(loss)
    # print(loss)
    return loss


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

# ##################Loss for matching text-image###################

def sent_loss(cnn_code, rnn_code, labels, class_ids,
              batch_size, cfg, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels,
               cap_lens, class_ids, batch_size, cfg):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    words_num = cfg.TEXT.WORDS_NUM
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        # words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


# ##################Loss for G and Ds##############################
def discriminator_loss(netD, real_imgs, fake_imgs, conditions,
                       real_labels, fake_labels, writer, i, count):
    # Forward
    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())
    # loss
    #
    cond_real_logits = netD.COND_DNET(real_features, conditions).mean()
    cond_fake_logits = netD.COND_DNET(fake_features, conditions).mean()
    #
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size]).mean()
    if netD.UNCOND_DNET is not None:
        real_logits = netD.UNCOND_DNET(real_features).mean()
        fake_logits = netD.UNCOND_DNET(fake_features).mean()
        errD = (fake_logits - real_logits + cond_fake_logits + cond_wrong_logits - cond_real_logits)
        writer.add_scalar("Dis_%d/real_err" % i, -real_logits.data.cpu().numpy(), count)
        writer.add_scalar("Dis_%d/fake_err" % i, fake_logits.data.cpu().numpy(), count)
        writer.add_scalar("Dis_%d/cond_real_err" % i, -cond_real_logits.data.cpu().numpy(), count)
        writer.add_scalar("Dis_%d/cond_fake_err" % i, cond_fake_logits.data.cpu().numpy(), count)
        writer.add_scalar("Dis_%d/cond_wrong_err" % i, cond_wrong_logits.data.cpu().numpy(), count)
    else:
        pass
        # errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.
        # writer.add_scalar("Dis_%d/cond_real_err" % i, cond_real_errD.data.cpu().numpy(), count)
        # writer.add_scalar("Dis_%d/cond_fake_err" % i, cond_fake_errD.data.cpu().numpy(), count)
        # writer.add_scalar("Dis_%d/cond_wrong_err" % i, cond_wrong_errD.data.cpu().numpy(), count)

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_imgs)
    alpha = alpha.cuda()
    interpolated = alpha * real_imgs + (1 - alpha) * fake_imgs
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(
                               prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    writer.add_scalar("Dis_%d/gradient_penalty" % i, gradient_penalty.data.cpu().numpy(), count)
    errD += gradient_penalty * 10

    return errD


def generator_loss(netsD, image_encoder, real_imgs, fake_imgs, real_labels,
                   words_embs, sent_emb, match_labels,
                   cap_lens, class_ids, writer, count, cfg):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = '|'
    # Forward
    errG_total = 0
    for i in range(numDs):
        features = netsD[i](fake_imgs[i])
        cond_logits = netsD[i].COND_DNET(features, sent_emb).mean()
        writer.add_scalar("Gen_%d/cond_err" % i, -cond_logits.data.cpu().numpy(), count)
        if netsD[i].UNCOND_DNET is not None:
            logits = netsD[i].UNCOND_DNET(features).mean()
            writer.add_scalar("Gen_%d/err" % i, -logits.data.cpu().numpy(), count)
            g_loss = -logits - cond_logits
        else:
            g_loss = -cond_logits
        errG_total += g_loss / 1000
        # err_img = errG_total.data[0]
        logs += ' g_loss%d: %.2f |' % (i, g_loss.data.cpu().numpy())

        # Ranking loss
        if i == (numDs - 1) or (cfg.TRAIN.SECOND_LIMIT and i == (numDs -2)):
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef
            region_features, cnn_code, fake_id_features, fake_attribute = image_encoder(fake_imgs[i])
            _, _, real_id_features, real_attribute = image_encoder(real_imgs)
            id_loss = 1 - torch.mean(cosine_similarity(fake_id_features, real_id_features, dim=1))
            writer.add_scalar("Gen_%d/id_loss" % i, id_loss.data, count)
            logs += ' id_loss_%d: %.2f |' % (i, id_loss.data.cpu().numpy())
            errG_total += id_loss * cfg.TRAIN.SMOOTH.LAMBDA2

            attr_loss = multi_binary_cross_entropy(fake_attribute, real_attribute)
            writer.add_scalar("Gen_%d/attr_loss" % i, attr_loss.data, count)
            logs += ' attr_loss_%d: %.2f |' % (i, attr_loss.data.cpu().numpy())
            errG_total += attr_loss * cfg.TRAIN.SMOOTH.LAMBDA3

            w_loss0, w_loss1, _ = words_loss(region_features, words_embs,
                                             match_labels, cap_lens,
                                             class_ids, batch_size, cfg)
            w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
            # err_words = err_words + w_loss.data[0]

            s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb,
                                         match_labels, class_ids, batch_size, cfg)
            s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
            # err_sent = err_sent + s_loss.data[0]
            writer.add_scalar("Gen_%d/word_loss0" % i, w_loss0.data, count)
            writer.add_scalar("Gen_%d/word_loss1" % i, w_loss1.data, count)
            writer.add_scalar("Gen_%d/sentence_loss0" % i, s_loss0.data, count)
            writer.add_scalar("Gen_%d/sentence_loss1" % i, s_loss1.data, count)
            errG_total += w_loss + s_loss
            logs += ' w_loss_%d: %.2f | s_loss_%d: %.2f |' % (i, w_loss.data.cpu().numpy(), i, s_loss.data.cpu().numpy())
    return errG_total, logs


##################################################################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

