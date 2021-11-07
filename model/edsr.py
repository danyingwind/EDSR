from model import common

import torch.nn as nn

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

# 这里需要args作为参数
def make_model(args, parent=False):
    return EDSR(args)

# 这里需要args，还需要common.py提供的default_conv函数
class EDSR(nn.Module):
    # 定义模型的各个组件==
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()
        n_resblocks = args.n_resblocks # 来自args的参数
        n_feats = args.n_feats # 来自args的参数
        kernel_size = 3  # 自定义参数
        scale = args.scale[0] # 来自args的参数
        act = nn.ReLU(True)

        # 这里还不太明白url的作用是什么.用于指定下载模型的位置
        # 这里只会对中括号进行内容替换，'r''f''x'都会保留
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale) 
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None

        # 来自common.py中定义的函数
        self.sub_mean = common.MeanShift(args.rgb_range) 
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)



        #==========================================开始定义模型==========================================#

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        # 这里是给各个模块进行重命名
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        #================================================模型结束=============================================#
    # 定义模型的前向传播流程
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

