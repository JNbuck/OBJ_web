#                    .::::.
#                  .::::::::.
#                 :::::::::::
#              ..:::::::::::'
#           '::::::::::::'
#             .::::::::::
#        '::::::::::::::..
#             ..::::::::::::.
#           ``::::::::::::::::
#            ::::``:::::::::'        .:::.
#           ::::'   ':::::'       .::::::::.
#         .::::'      ::::     .:::::::'::::.
#        .:::'       :::::  .:::::::::' ':::::.
#       .::'        :::::.:::::::::'      ':::::.
#      .::'         ::::::::::::::'         ``::::.
#  ...:::           ::::::::::::'              ``::.
# ````':.          ':::::::::'                  ::::..
#                    '.:::::'                    ':'````..

# !/usr/bin/env/python
# encoding:utf-8
# author: usr
from utils import Detection
import cv2
import os
import time
import torch

from nanodet.util import Logger
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.data.transform import Pipeline
from OBJ_web.config.config import cfg, load_config


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device='cuda:0'):
        self.cfg = cfg  # 配置文件
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {}
        if isinstance(img, str):
            img_info['file_name'] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info['file_name'] = None

        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        meta = dict(img_info=img_info,
                    raw_img=img,
                    img=img)
        meta = self.pipeline(meta, self.cfg.data.val.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result = self.model.head.show_result(meta['raw_img'], dets, class_names, score_thres=score_thres, show=True)
        print('viz time: {:.3f}s'.format(time.time()-time1))
        return result

    def denumber(self, dets, score_thresh=0.35):
        persones = 0
        cars = 0
        for index in (0, 1, 2, 3, 5, 7):
            for bbox in dets[index]:
                score = bbox[-1]
                if score > score_thresh and index == 0:
                    persones += 1
                elif score > score_thresh and index in (1, 2, 3, 5, 7):
                    cars += 1
        print('persones: {}'.format(persones), end=' | ')
        print('car: {}'.format(cars), end=' | ')
        count = {'persones': persones, 'cars': cars}
        return count

    def decars(self, dets, score_thresh=0.35):
        cars = 0
        for index in (1, 2, 3, 5, 7):
            for bbox in dets[index]:
                score = bbox[-1]
                if score > score_thresh:
                    cars += 1
        print('car: {}'.format(cars), end=' | ')
        return cars

    def depersones(self, dets, score_thresh=0.35):
        persones = 0
        for bbox in dets[0]:
            score = bbox[-1]
            if score > score_thresh:
                persones += 1
        print('persones: {}'.format(persones), end=' | ')
        return persones


class Nanodet_Detection(Detection):

    def __init__(self, req):

        # 运行cuDNN使用非确定算法，允许cuDNN自动寻找最适合当前配置的高效算法
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        yaml_path = "./config/config.yaml"
        model_path = "./model/nanodet_m.pth"
        # 初始化配置文件（将配置文件导入程序变量）
        load_config(cfg, yaml_path)
        # 初始化日志对象
        self.logger = Logger(-1, use_tensorboard=False)
        # 初始化模型对象
        predictor = Predictor(cfg, model_path, self.logger, device='cuda:0')
        self.logger.log('Press "Esc", "q" or "Q" to exit.')
        Detection.__init__(self, req)
        self.predictor = predictor

    def detection(self):
        # 超参数：置信度，超过此得分才能算识别成功
        score_thresh = 0.3
        # result = {'sty': "nanodet_success"}
        result = {}
        img = self.openImage()
        cv2.imwrite("test_post.jpg", img)
        meta, res = self.predictor.inference(img)
        cars = self.predictor.decars(res, score_thresh=score_thresh)  # 测试
        result["cars"] = cars
        self.logger.log("car: " + str(cars))
        res_img = self.predictor.visualize(res, meta, cfg.class_names, score_thres=score_thresh)

        # ch = cv2.waitKey(0)
        # if ch == 27 or ch == ord('q') or ch == ord('Q'):
        #     cv2.imwrite("test.jpg", res_img)
        #     return result
        cv2.imwrite("test.jpg", res_img)
        return result
