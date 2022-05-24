import torch
from torch import nn

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
        
        
class Model_1(nn.Sequential): # 984 params
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(        
            nn.Conv2d(3, 4, kernel_size=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), # 16x16
            
            nn.Conv2d(4, 8, kernel_size=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=4),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), # 8x8
            
            nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), # 4x4
            
            nn.Conv2d(8, 4, kernel_size=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            
            torch.nn.AvgPool2d(kernel_size=4),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 100),
            torch.nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.layers(x)

        
class Model_2(nn.Sequential): # 4,956 params (ex Model9)
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(        
            nn.Conv2d(3, 4, kernel_size=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), # 16x16
            
            nn.Conv2d(4, 8, kernel_size=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=4),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), # 8x8
            
            nn.Conv2d(8, 16, kernel_size=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), # 4x4
            
            nn.Conv2d(16, 32, kernel_size=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            torch.nn.AvgPool2d(kernel_size=4),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 100),
            torch.nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.layers(x)

    
class Model_3(nn.Sequential): # 9,636 params (ex Model8) maybe change it?
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(        
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), # 16x16
            
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), # 8x8
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), # 4x4
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            torch.nn.AvgPool2d(kernel_size=4),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 100),
            torch.nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    
class Model_4(nn.Sequential): # 50,548 params (ex Model6)
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        
            MBConv(in_c=16, out_c=16, inner_c=64), # 32x32
            
            nn.MaxPool2d(kernel_size=2),
            MBConv(in_c=16, out_c=32, inner_c=64), # 16x16
            
            nn.MaxPool2d(kernel_size=2),
            MBConv(in_c=32, out_c=32, inner_c=128), # 8x8
            
            nn.MaxPool2d(kernel_size=2),
            MBConv(in_c=32, out_c=100, inner_c=128), # 4x4
            
            nn.AvgPool2d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(100, 100),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.layers(x)

    
class Model_5(nn.Sequential): # 99,844 params (ex Model5m)
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            MBConv(in_c=32, out_c=32, inner_c=128), # 32x32
            
            nn.MaxPool2d(kernel_size=2),
            MBConv(in_c=32, out_c=32, inner_c=128), # 16x16
            
            nn.MaxPool2d(kernel_size=2),
            MBConv(in_c=32, out_c=64, inner_c=128), # 8x8
            
            nn.MaxPool2d(kernel_size=2),
            MBConv(in_c=64, out_c=100, inner_c=256), # 4x4
            
            nn.AvgPool2d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(100, 100),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.layers(x)
    
        
class Model_6(nn.Sequential): # 352,388 params (ex Model2)
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            MBConv(in_c=32, out_c=32, inner_c=128), # 32x32
            
            nn.MaxPool2d(kernel_size=2),
            MBConv(in_c=32, out_c=64, inner_c=128), # 16x16
            
            nn.MaxPool2d(kernel_size=2),
            MBConv(in_c=64, out_c=128, inner_c=256), # 8x8
            
            nn.MaxPool2d(kernel_size=2),
            MBConv(in_c=128, out_c=256, inner_c=512), # 4x4
            
            nn.AvgPool2d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(256, 100),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.layers(x)
    

class Model_7(nn.Sequential): # 491,524 params (ex. model 3)
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            MBConv(in_c=32, out_c=32, inner_c=128), # 32x32
            
            nn.MaxPool2d(kernel_size=2),
            MBConv(in_c=32, out_c=64, inner_c=128), # 16x16
            
            nn.MaxPool2d(kernel_size=2),
            MBConv(in_c=64, out_c=128, inner_c=256), # 8x8
            
            MBConv(in_c=128, out_c=128, inner_c=512), # 8x8
            
            nn.MaxPool2d(kernel_size=2),
            MBConv(in_c=128, out_c=256, inner_c=512), # 4x4
            
            nn.AvgPool2d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(256, 100),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    
class Model_8(nn.Sequential): # 1,068,740 params (ex Model4)
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            MBConv(in_c=32, out_c=32, inner_c=128), # 32x32
            
            nn.MaxPool2d(kernel_size=2),
            MBConv(in_c=32, out_c=64, inner_c=128), # 16x16
            MBConv(in_c=64, out_c=64, inner_c=256), # 16x16
            
            nn.MaxPool2d(kernel_size=2),
            MBConv(in_c=64, out_c=128, inner_c=256), # 8x8
            MBConv(in_c=128, out_c=128, inner_c=512), # 8x8
            
            nn.MaxPool2d(kernel_size=2),
            MBConv(in_c=128, out_c=256, inner_c=512), # 4x4
            MBConv(in_c=256, out_c=256, inner_c=1024), # 4x4
            
            nn.AvgPool2d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(256, 100),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)