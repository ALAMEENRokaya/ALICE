import torch
import torch.nn as nn

class FCWrapper(nn.Module):
    def __init__(self,FC:nn.Linear,alpha=8,r=8):
        super().__init__()
        self.fc=FC
        self.wa=nn.Linear(self.fc.in_features,r)
        self.wb=nn.Linear(r,self.fc.out_features)
        nn.init.zeros_(self.wa.weight)
        self.alpha=alpha
        nn.init.normal_(self.wb.weight,0,1)
    def forward(self,x):
        adapter=self.alpha*self.wb(self.wa(x))
        adapted_output=adapter+self.fc(x)
        return adapted_output
