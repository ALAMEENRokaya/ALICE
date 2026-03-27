import torch
import torch.nn as nn

class FCWrapper(nn.Module):
    def __init__(self,FC:nn.Linear,alpha=8,r=8):
        super().__init__()
        self.fc=FC
        self.wa=nn.Linear(self.fc.in_features,r)
        self.wb=nn.Linear(r,self.fc.out_features)
        nn.init.zeros_(self.wa.weight)
        nn.init.zeros_(self.wa.bias)
        self.alpha=alpha
        nn.init.normal_(self.wb.weight,0,1)
        nn.init.zeros_(self.wb.bias)
        device=FC.weight.device
        self.wa = self.wa.to(device)
        self.scale = alpha/r 
        self.wb = self.wb.to(device)
    def forward(self,x):
        adapter=self.scale*self.wb(self.wa(x))
        adapted_output=adapter+self.fc(x)
        return adapted_output
    
def wrap_fc_layers(model, alpha=8, r=8):
    for name, module in model.named_modules():
        if ('layers' in name or 'syn_layers' in name) and type(module).__name__=='Mlp':
            module.fc1 = FCWrapper(module.fc1, alpha, r)
            module.fc2 = FCWrapper(module.fc2, alpha, r)           
    return model

