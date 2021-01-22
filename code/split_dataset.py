# -*- coding: utf-8 -*-
# @Time    : 2019/2/20 11:15
# @Author  : Yuan Zheng
# @Version : 0.0.1
# @File    : split_dataset.py

import pickle
import click
import glob
import os
from random import shuffle

@click.command()
@click.option('--img_path',
              type=click.STRING)
@click.option('--attr_path',
              type=click.STRING)
@click.option('--dir',
              type=click.STRING,
              default='CelebA')
def main(img_path, attr_path, dir):
    delete_list = []
    d = dict()
    imgs = glob.glob(img_path+'/*.jpg')
    for img in imgs:
        base = os.path.basename(img)
        d[base] = [img]
    identity_path = os.path.join(attr_path, "identity_CelebA.txt")
    with open(identity_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            img, id = line.split()
            id = int(id)
            if img in d:
                d[img].append(id)
    attr_path = os.path.join(attr_path, "list_attr_celeba.txt")
    with open(attr_path, 'r') as file:
        lines = file.readlines()
        attrs = lines[1]
        attrs = attrs.split()
        num2attr = dict()
        attr2num = dict()
        redirect = dict()
        attr_num = 0
        del_list_num = []
        for i, attr in enumerate(attrs):
            if i not in delete_list:
                redirect[i] = attr_num
                num2attr[attr_num] = attr
                attr2num[attr] = attr_num
                attr_num += 1
            else:
                del_list_num.append(i)
        for line in lines[2:]:
            items = line.split()
            img = items[0]
            attrs = [redirect[i] for i, item in enumerate(items[1:]) if item == "1" and i not in del_list_num]
            if img in d:
                if len(attrs) > 0:
                    d[img].append(attrs)
                else:
                    d.pop(img)
            # if img in check_list:
            #     print(img, attrs)
    # exit()
    print("Attribute num: ", attr_num)
    if not os.path.isdir("../data/"+dir):
        os.makedirs("../data/"+dir)
    dict_path = "../data/"+ dir +"/dict.pickle"
    with open(dict_path, 'wb') as f:
        pickle.dump([num2attr, attr2num], f, protocol=2)
    filename = list(d.keys())
    shuffle(filename)
    total_num = len(filename)
    train_num = int(total_num * 0.9)
    test_num = total_num - train_num
    print("Total: %d  Train: %d  Test: %d" % (total_num, train_num, test_num))
    train_filename = filename[:train_num]
    train_filepath = [d[filename][0] for filename in train_filename]
    train_id = [d[filename][1] for filename in train_filename]
    train_attr = [d[filename][2] for filename in train_filename]

    test_filename = filename[train_num:]
    test_filepath = [d[filename][0] for filename in test_filename]
    test_id = [d[filename][1] for filename in test_filename]
    test_attr = [d[filename][2] for filename in test_filename]

    train_path = "../data/" + dir + "/train.pickle"
    with open(train_path, 'wb') as f:
        pickle.dump([train_filename, train_filepath, train_id, train_attr], f, protocol=2)

    test_path = "../data/" + dir + "/test.pickle"
    with open(test_path, 'wb') as f:
        pickle.dump([test_filename, test_filepath, test_id, test_attr], f, protocol=2)
if __name__ == '__main__':
    main()