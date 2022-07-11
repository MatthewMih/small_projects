import torch
from torch import nn
from collections import OrderedDict

class Parallel(nn.Module): # штучка для residual connection'ов
    def __init__(self, block1, block2): # ко входному тензору примерняются block1, block2 и складываются рез-таты. 
        super().__init__()
        self.block1 = block1
        self.block2 = block2
        
    def forward(self, x):
        return self.block1(x) + self.block2(x)
    

class MBConv(nn.Sequential):
    def __init__(self, in_c, out_c, inner_c):
        super().__init__(
            Parallel(
                nn.Sequential(
                    nn.Conv2d(in_c, inner_c, kernel_size=1),
                    nn.BatchNorm2d(inner_c),
                    nn.ReLU(),
                    
                    nn.Conv2d(inner_c, inner_c, kernel_size=3, groups=inner_c, padding=1),
                    nn.BatchNorm2d(inner_c),
                    nn.ReLU(),
                    
                    nn.Conv2d(inner_c, out_c, kernel_size=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(),
                ),
                nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, kernel_size=1)
            )
        )


class MBConvs_block(nn.Sequential):
    def __init__(self, k, in_c, out_c, inner_c):
        super().__init__()
        MBConvs = []
        
        if k == 0:
            MBConvs.append(('conv', nn.Conv2d(in_c, out_c, kernel_size=1)))
            MBConvs.append(('bn',nn.BatchNorm2d(out_c)))
            MBConvs.append(('relu',nn.ReLU()))
        else:
            MBConvs.append(('mbconv1', MBConv(in_c, out_c, inner_c)))
            for i in range(k-1):
                MBConvs.append((f'mbconv{i+2}',MBConv(out_c, out_c, inner_c)))
        
        self.MBConvs = torch.nn.Sequential(OrderedDict(MBConvs))
    
    def forward(self, x):
        return self.MBConvs(x)


class Architecture_params:
    def __init__(self, k1, c_out1, c_inner1, k2, c_out2, c_inner2, k3, c_out3, c_inner3, k4, c_out4, c_inner4):
        self.k1 = k1
        self.c_out1 = c_out1
        self.c_inner1 = c_inner1
        
        self.k2 = k2
        self.c_out2 = c_out2
        self.c_inner2 = c_inner2
        
        self.k3 = k3
        self.c_out3 = c_out3
        self.c_inner3 = c_inner3
        
        self.k4 = k4
        self.c_out4 = c_out4
        self.c_inner4 = c_inner4


class Model(torch.nn.Module):
    def __init__(self, architecture_params: Architecture_params):
        super().__init__()
        
        self.architecture_params = architecture_params
        self.layers = nn.Sequential(
            MBConvs_block(k=architecture_params.k1, in_c=3,
                          out_c=architecture_params.c_out1,
                          inner_c=architecture_params.c_inner1),
            nn.MaxPool2d(kernel_size=2), # 32x32 -> 16x16
            
            MBConvs_block(k=architecture_params.k2, in_c=architecture_params.c_out1,
                          out_c=architecture_params.c_out2,
                          inner_c=architecture_params.c_inner2),
            nn.MaxPool2d(kernel_size=2), # 16x16 -> 8x8
            
            MBConvs_block(k=architecture_params.k3, in_c=architecture_params.c_out2,
                          out_c=architecture_params.c_out3,
                          inner_c=architecture_params.c_inner3),
            nn.MaxPool2d(kernel_size=2), # 8x8 -> 4x4
            
            MBConvs_block(k=architecture_params.k4, in_c=architecture_params.c_out3,
                          out_c=architecture_params.c_out4,
                          inner_c=architecture_params.c_inner4),
            
            nn.AvgPool2d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(architecture_params.c_out4, 100),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)