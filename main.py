import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)#为GPU设置随机种子
checkpoint = utility.checkpoint(args)#处理内存相关问题

def main():
    global model
    if args.data_test == ['video']: # 运行视频测试模块
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else: # 运行训练模块
        if checkpoint.ok: #检查参数
            loader = data.Data(args) # 提供一些参数
            _model = model.Model(args, checkpoint) # 提供一些参数
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None # 提供一些参数选择
            t = Trainer(args, loader, _model, _loss, checkpoint) # 初始化训练模块
            while not t.terminate(): # 根据训练/测试模式及epoch，确定训练次数
                t.train()
                t.test()

            checkpoint.done() # 关闭日志文件

if __name__ == '__main__':
    main()
