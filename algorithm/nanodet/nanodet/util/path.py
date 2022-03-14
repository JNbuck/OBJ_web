import os
from .rank_filter import rank_filter


# @关键符的作用在于将@下函数作为一个变量传入@后面这个函数之中
@rank_filter
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
