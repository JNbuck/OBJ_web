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
import cv2
import numpy as np
import base64


class Detection():

    def __init__(self, msg, jsonName='image'):
        self.msg = msg
        self.jsonName = jsonName

    def detection(self):
        raise NotImplementedError

    def demo_test(self):
        result = {'sty': "success"}

        img = self.openImage()
        cv2.imwrite("test.jpg", img)

        return result

    def openImage(self):

        img_decode_as = self.msg[self.jsonName].encode('ascii')     # 从unicode变成ascii编码
        img_decode = base64.b64decode(img_decode_as)    # 解base64编码，得图片的二进制
        img_np_ = np.frombuffer(img_decode, np.uint8)
        img = cv2.imdecode(img_np_, cv2.COLOR_RGB2BGR)  # 转为opencv格式

        return img