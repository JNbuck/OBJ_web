

# 接收传入的函数作为参数，（*args,**kwargs）是传入函数的参数
# 以下的这种写法避免了编译阶段的调用，只有在调用使用了这个@的函数的时候才会调用下面的这个函数
def rank_filter(func):
    def func_filter(local_rank=-1, *args, **kwargs):
        if local_rank < 1:
            return func(*args, **kwargs)
        else:
            pass
    return func_filter
