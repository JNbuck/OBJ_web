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
from .yacs import CfgNode
# 用来将配置文件载入程序，并将每个节点都变成变量

cfg = CfgNode(new_allowed=True)
# 算法配置
cfg.algorithm = CfgNode(new_allowed=True)

# common params for NETWORK
cfg.model = CfgNode()
cfg.model.arch = CfgNode(new_allowed=True)
cfg.model.arch.backbone = CfgNode(new_allowed=True)
cfg.model.arch.neck = CfgNode(new_allowed=True)
cfg.model.arch.head = CfgNode(new_allowed=True)

# DATASET related params
cfg.data = CfgNode(new_allowed=True)
cfg.data.train = CfgNode(new_allowed=True)
cfg.data.val = CfgNode(new_allowed=True)
cfg.device = CfgNode(new_allowed=True)
# train
cfg.schedule = CfgNode(new_allowed=True)

# logger
cfg.log = CfgNode()
cfg.log.interval = 50

# testing
cfg.test = CfgNode()
# size of images for each device

def load_config(cfg, args_cfg):
    # 使得该父节点和其子节点都是可变的，由于cfg是最大的父节点，因此该配置文件的所有节点可变
    cfg.defrost()
    # 通过yml传入参数，自动将子节点变量等一系列变量通过配置文件补齐
    cfg.merge_from_file(args_cfg)
    # 冻结配置文件变量，防止后续被修改
    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(cfg, file=f)
