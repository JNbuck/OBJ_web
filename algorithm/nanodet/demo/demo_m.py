import datetime
import socket

import cv2
import os
import time

import numpy as np
import torch
import argparse
from PIL import Image
from io import BytesIO
from OBJ_web.algorithm.nanodet.nanodet.util import cfg, load_config, Logger
from OBJ_web.algorithm.nanodet.nanodet.model.arch import build_model
from OBJ_web.algorithm.nanodet.nanodet.util import load_model_weight
from OBJ_web.algorithm.nanodet.nanodet.data.transform import Pipeline
from subsystem.database import Database




image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png']
video_ext = ['mp4', 'mov', 'avi', 'mkv']


# 编辑命令行输入的参数类型，设置提示信息等功能，异常实用
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('demo', default='image', help='demo type, eg. image, video , webcam_url and webcam')
    parser.add_argument('--config', help='model config file path')
    parser.add_argument('--model', help='model file path')
    parser.add_argument('--path', default='./demo', help='path to images or video')
    parser.add_argument('--url', help='webcam_url demo url')
    parser.add_argument('--post', help='webcam_url demo post')
    parser.add_argument('--camid', type=int, default=0, help='webcam demo camera id')
    args = parser.parse_args()
    return args


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
        self.model.head.show_result(meta['raw_img'], dets, class_names, score_thres=score_thres, show=True)
        print('viz time: {:.3f}s'.format(time.time()-time1))

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



def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names


def main():
    args = parse_args()
    # 运行cuDNN使用非确定算法，允许cuDNN自动寻找最适合当前配置的高效算法
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # 初始化配置文件（将配置文件导入程序变量）
    load_config(cfg, args.config)
    # 初始化日志对象
    logger = Logger(-1, use_tensorboard=False)
    # 初始化模型对象
    predictor = Predictor(cfg, args.model, logger, device='cuda:0')
    logger.log('Press "Esc", "q" or "Q" to exit.')
    if args.demo == 'image':
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()
        for image_name in files:
            meta, res = predictor.inference(image_name)
            # predictor.denumber(res)  # 测试
            predictor.visualize(res, meta, cfg.class_names, 0.35)
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break
    elif args.demo == 'video' or args.demo == 'webcam':
        cap = cv2.VideoCapture(args.path if args.demo == 'video' else args.camid)
        while True:
            ret_val, frame = cap.read()
            meta, res = predictor.inference(frame)
            # predictor.denumber(res)  # 测试
            predictor.visualize(res, meta, cfg.class_names, 0.35)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break
    elif args.demo == 'webcam_url':
        # 注意IP地址和端口号与前面的程序中的保持一致
        HOST, PORT = args.url, int(args.post)
        # 连接到服务器
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        sign = True
        database = Database(key="test")
        database.connection()
        max_cars = 0
        max_persones = 0
        now_time = int(datetime.datetime.now().timestamp())
        while True:
            msg = sock.recv(1024 * 1024)
            if not msg:
                break
            buf = BytesIO(msg)
            buf.seek(0)
            try:
                t1 = int(datetime.datetime.now().timestamp())
                pi = Image.open(buf)  # 使用PIL读取jpeg图像数据

                img_array = np.array(pi)
                if img_array.dtype == object:
                    continue
                    # img_array = img_array.astype(np.uint8)
                # print(img_array.dtype)
                frame = cv2.cvtColor(np.asarray(img_array), cv2.COLOR_RGB2BGR)
                meta, res = predictor.inference(frame)
                if sign:
                    cars = predictor.decars(res)  # 测试
                    sign = False
                    max_cars = cars
                else:
                    persones = predictor.depersones(res)  # 测试
                    sign = True
                    max_persones = persones
                if t1 >= now_time+1000:
                    now_time = t1
                    database.insert(t1, 2, max_cars)
                    database.insert(t1, 1, max_persones)
                    max_cars, max_persones = 0, 0
                    database.commit()
                predictor.visualize(res, meta, cfg.class_names, 0.35)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord('q') or ch == ord('Q'):
                    database.close()
                    break

            except(OSError, NameError, ValueError):
                print('OSError')

            if cv2.waitKey(10) == ord('q'):
                break

        sock.close()
        cv2.destroyAllWindows()

    elif args.demo == 'webcam_p':
        # 注意IP地址和端口号与前面的程序中的保持一致
        HOST, PORT = args.url, int(args.post)
        # 连接到服务器
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        database = Database(key="test")
        database.connection()
        max_persones = 1
        now_time = 0
        while True:
            msg = sock.recv(1024 * 1024)
            if not msg:
                break
            buf = BytesIO(msg)
            buf.seek(0)
            try:
                t1 = int(datetime.datetime.now().timestamp())
                pi = Image.open(buf)  # 使用PIL读取jpeg图像数据

                img_array = np.array(pi)
                if img_array.dtype == object:
                    continue
                    # img_array = img_array.astype(np.uint8)
                # print(img_array.dtype)
                frame = cv2.cvtColor(np.asarray(img_array), cv2.COLOR_RGB2BGR)
                meta, res = predictor.inference(frame)
                persones = predictor.depersones(res)  # 测试
                if max_persones < persones:
                    max_persones = persones
                if t1 >= int(now_time+1):
                    now_time = t1
                    database.insert(t1, 1, max_persones)
                    max_persones = 1
                    database.commit()
                predictor.visualize(res, meta, cfg.class_names, 0.35)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord('q') or ch == ord('Q'):
                    database.close()
                    break

            except(OSError, NameError, ValueError):
                print('OSError')

            if cv2.waitKey(10) == ord('q'):
                break

        sock.close()
        cv2.destroyAllWindows()

    elif args.demo == 'webcam_c':
        # 注意IP地址和端口号与前面的程序中的保持一致
        HOST, PORT = args.url, int(args.post)
        # 连接到服务器
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        database = Database(key="test")
        database.connection()
        max_cars = 1
        now_time = int(datetime.datetime.now().timestamp())
        while True:
            msg = sock.recv(1024 * 1024)
            if not msg:
                break
            buf = BytesIO(msg)
            buf.seek(0)
            try:
                t1 = int(datetime.datetime.now().timestamp())
                pi = Image.open(buf)  # 使用PIL读取jpeg图像数据

                img_array = np.array(pi)
                if img_array.dtype == object:
                    continue
                    # img_array = img_array.astype(np.uint8)
                # print(img_array.dtype)
                frame = cv2.cvtColor(np.asarray(img_array), cv2.COLOR_RGB2BGR)
                meta, res = predictor.inference(frame)
                cars = predictor.decars(res)  # 测试
                if max_cars < cars:
                    max_cars = cars
                if t1 >= now_time+1:
                    now_time = t1
                    database.insert(t1, 2, max_cars)
                    max_cars = 1
                    database.commit()
                predictor.visualize(res, meta, cfg.class_names, 0.35)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord('q') or ch == ord('Q'):
                    database.close()
                    break

            except(OSError, NameError, ValueError):
                print('OSError')

            if cv2.waitKey(10) == ord('q'):
                break

        sock.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()