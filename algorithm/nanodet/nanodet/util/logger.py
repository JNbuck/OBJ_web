import os
import logging
import torch
import numpy as np
from termcolor import colored
from .rank_filter import rank_filter
from .path import mkdir


class Logger:
    def __init__(self, local_rank, save_dir='./', use_tensorboard=True):
        # 注意是在你执行该命令的那个目录创建一个目录
        mkdir(local_rank, save_dir)
        self.rank = local_rank
        fmt = colored('[%(name)s]', 'magenta', attrs=['bold']) + colored('[%(asctime)s]', 'blue') + \
              colored('%(levelname)s:', 'green') + colored('%(message)s', 'white')
        # 配置logging，指定输出的日志等级，输出的文件名，打开文件的权限
        logging.basicConfig(level=logging.INFO,
                            filename=os.path.join(save_dir, 'logs.txt'),
                            filemode='w')
        # 建立保存tensorborad的数据文件
        self.log_dir = os.path.join(save_dir, 'logs')
        # 指定日志的输出流格式
        console = logging.StreamHandler()
        # 指定输出流的数据等级
        console.setLevel(logging.INFO)
        # 拼接好输出流的格式形式
        formatter = logging.Formatter(fmt, datefmt="%m-%d %H:%M:%S")
        # 指定输出流的格式内容
        console.setFormatter(formatter)
        # 添加设置好的数据流
        logging.getLogger().addHandler(console)

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')
            if self.rank < 1:
                logging.info('Using Tensorboard, logs will be saved in {}'.format(self.log_dir))
                self.writer = SummaryWriter(log_dir=self.log_dir)

    def log(self, string):
        if self.rank < 1:
            logging.info(string)

    def scalar_summary(self, tag, phase, value, step):
        if self.rank < 1:
            self.writer.add_scalars(tag, {phase: value}, step)


class MovingAverage(object):
    def __init__(self, val, window_size=50):
        self.window_size = window_size
        self.reset()
        self.push(val)

    def reset(self):
        self.queue = []

    def push(self, val):
        self.queue.append(val)
        if len(self.queue) > self.window_size:
            self.queue.pop(0)

    def avg(self):
        return np.mean(self.queue)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, val):
        self.reset()
        self.update(val)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
