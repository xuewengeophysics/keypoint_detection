#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 转化成适合的格式便于测试mAP准确率
import os
import json
import numpy as np
import ipdb;pdb=ipdb.set_trace
import shutil
from argparse import ArgumentParser

np.random.seed(10101)
def find_items(images, anns):
    lists = []
    for img in images:
        image_id = img['id']
        for ann in anns:
            if image_id == ann['image_id']:
                lists.append(ann)
    return lists

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--ants_path", help="input file path", default="./data/coco/annotations/person_keypoints_val2017.json")  # 通过-p指定路径，标注文件名
    args = parser.parse_args()
    ants_path = args.ants_path
    '''
    with open("../shape/annotations/instances_train.json", "rt") as f:
        data = json.loads(f.read())
    print(data.keys())

    '''
    with open(ants_path, "rt") as f:
        data = eval(f.read())
    print('data.keys() = ', data.keys())
    print('info = ', data['info'])
    print('licenses num = ', len(data['licenses']))
    # for i in range(len(data['licenses'])):
    #     print('licenses = ', data['licenses'][i])
    print('images num = ', len(data['images']))
    print('images = ', data['images'][0])

    print('annotations num = ', len(data['annotations']))
    print('annotations = ', data['annotations'][0])

    print('categories num = ', len(data['categories']))
    print('categories = ', data['categories'][0])

    # data = data['annotations']
    # anns = []
    # id_num = 0
    # image_id_num = 0
    # images = []
    # for dd in data:
    #     for d in dd[1:]:
    #         x1, y1, x2, y2 = d[:4]
    #         w, h = x2-x1, y2-y1
    #         bbox = [x1, y1, w, h]
    #         area = w*h
    #         image_name = dd[0]
    #         image_id = image_id_num
    #         id = id_num
    #         id_num += 1
    #         item = {"id": id, "image_id": image_id, "category_id": 1, "iscrowd": 0, 'area': area, "bbox": bbox}
    #         anns.append(item)
    #     img = {"id": image_id, "file_name": image_name}
    #     images.append(img)
    #     image_id_num += 1

    # np.random.shuffle(images)
    # len_val = int(len(images)*0.1)

    # val_imgs = images[: len_val]
    # val_anns = find_items(val_imgs, anns)

    # train_imgs = images[len_val : ]
    # train_anns = find_items(train_imgs, anns)

    # images_path = "images/"
    # train_path = "../pig/train/"
    # if not os.path.exists(train_path):
    #     os.makedirs(train_path)
    # val_path = "../pig/val/"
    # if not os.path.exists(val_path):
    #     os.makedirs(val_path)
    # anns_path = "../pig/annotations/"
    # if not os.path.exists(anns_path):
    #     os.makedirs(anns_path)
    # for img in val_imgs:
    #     image_name = img['file_name']
    #     file_image = images_path + image_name
    #     shutil.copy(file_image, val_path)
    # for img in train_imgs:
    #     image_name = img['file_name']
    #     file_image = images_path + image_name
    #     shutil.copy(file_image, train_path)

    # with open(anns_path + 'instances_val.json', 'wt') as f:
    #     val_data = {"categories": [{"id": 1, "name": "pig", "supercategory": "None"}],
    #                 "images": val_imgs,
    #                 "annotations": val_anns}
    #     f.write(json.dumps(val_data))

    # with open(anns_path + 'instances_train.json', 'wt') as f:
    #     train_data = {"categories": [{"id": 1, "name": "pig", "supercategory": "None"}],
    #                   "images": train_imgs,
    #                   "annotations": train_anns}
    #     f.write(json.dumps(train_data))
