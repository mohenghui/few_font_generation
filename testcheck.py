from statistics import mode
import torch
import torch.nn as nn
from torchsummary import summary
class TestModule(nn.Module):
    def __init__(self):
        super(TestModule,self).__init__()
        self.layer1 = nn.Conv2d(16,32,3,1)
        self.layer2 = nn.Linear(32,10)
        # self.layer3 = nn.Linear(32,10)
        # self.layer3=nn.utils.spectral_norm()
 
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)

model = TestModule()

def add_sn(m):
        for name, layer in m.named_children():
            m.add_module(name, add_sn(layer))
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # print(1)
            return nn.utils.spectral_norm(m)
        else:
            print(m)
            return m
        
my_model = add_sn(model)
my_model.to('cuda')
summary(my_model,(16,128,128))

# for name, module in model.named_children():
#     print('children module:', name,module)

# for name, module in my_model.named_modules():
#     print('modules:', name,module)