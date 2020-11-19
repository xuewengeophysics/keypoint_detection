'''
使用yolov3作为pose net模型的前处理
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import ipdb;pdb=ipdb.set_trace
import numpy as np
from tqdm import tqdm
from keypoint_utils import plot_keypoint, preprocess

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
import config
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

from utils.transforms import *
from lib.core.inference import get_final_preds
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument("-i", "--img_input", help="input video file name", default='./data/coco/images/val2017/000000163682.jpg')
    parser.add_argument("-o", "--img_output", help="output video file name", default="output/result.png")
    parser.add_argument('--display', action='store_true')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


    ########## 加载human detector model
    from lib.detector.yolo.human_detector import load_model as yolo_model
    human_model = yolo_model()

    from lib.detector.yolo.human_detector import human_bbox_get as yolo_det
    print(args.img_input)
    img = cv2.imread(args.img_input)
    print(type(img))
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    bboxs, scores = yolo_det(args.img_input, human_model, confidence=0.5)  # bboxes (N, 4) [x0, y0, x1, y1]
    print("bboxs = ", bboxs)
    print("scores = ", scores)



    # bbox is coordinate location
    inputs, origin_img, center, scale = preprocess(args.img_input, bboxs, scores, cfg)
    print("inputs type = ", type(inputs))
    print("inputs shape after preprocess = ", inputs.shape)

    import onnxruntime
    import numpy as np
    sess = onnxruntime.InferenceSession("onnx_hrnet_human.onnx")

    input_name = sess.get_inputs()[0].name
    print("input name", input_name)
    input_shape = sess.get_inputs()[0].shape
    print("input shape", input_shape)
    input_type = sess.get_inputs()[0].type
    print("input type", input_type)

    output_name = sess.get_outputs()[0].name
    print("output name", output_name)
    output_shape = sess.get_outputs()[0].shape
    print("output shape", output_shape)
    output_type = sess.get_outputs()[0].type
    print("output type", output_type)

    # inference
    import numpy.random
    x = inputs.numpy()
    x = x.astype(numpy.float32)
    res = sess.run([output_name], {input_name: x})
    print("np.array(res[0]) shape = ", np.array(res[0]).shape)

    #ipdb.set_trace()

    preds, maxvals = get_final_preds(
        cfg, np.array(res[0]), np.asarray(center), np.asarray(scale))
    print("preds = ", preds)
    print("maxvals = ", maxvals)

    image = plot_keypoint(origin_img, preds, maxvals, 0.5)
    cv2.imwrite(args.img_output, image)
    #if args.display:
    #cv2.namedWindow("enhanced", cv2.WINDOW_GUI_NORMAL);
    #cv2.resizeWindow("enhanced", 960, 480);
    cv2.imshow('enhanced', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()