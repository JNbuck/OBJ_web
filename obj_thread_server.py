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
# -*- coding: utf-8 -*-
import codecs
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
# import DeviceFalutModels
import json
import sys
import utils
from config import cfg, load_config


sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# 服务端地址
host = ('localhost', 8000)


class Resquest(BaseHTTPRequestHandler):
    def handler(self):
        print("data:", self.rfile.readline().decode())
        self.wfile.write(self.rfile.readline())

    def do_GET(self):
        print(self.requestline)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        data = {"res": "success"}
        self.wfile.write(json.dumps(data).encode())

    # 接受post请求
    def do_POST(self):
        # 读取数据
        req_datas = self.rfile.read(int(self.headers['content-length']))

        req = json.loads(req_datas.decode())
        # 检测
        load_config(cfg, "./config/config.yaml")
        detector = utils.createInstance(cfg["algorithm"]["package"], cfg["algorithm"]["model"], req)
        result = detector.detection()

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        # 返回结果
        self.wfile.write(json.dumps(result).encode('utf-8'))


class ThreadingHttpServer(ThreadingMixIn, HTTPServer):
    pass


if __name__ == '__main__':
    myServer = ThreadingHttpServer(host, Resquest)

    print("Starting http server, listen at: %s:%s" % host)
    myServer.serve_forever()