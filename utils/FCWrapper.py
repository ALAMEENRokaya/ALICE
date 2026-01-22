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
        device = FC.weight.device
        self.wa = self.wa.to(device)
        self.wb = self.wb.to(device)
    def forward(self,x):
        adapter=self.alpha*self.wb(self.wa(x))
        adapted_output=adapter+self.fc(x)
        return adapted_output
    
def wrap_fc_layers(model, alpha=8, r=8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent = model
            attr_chain = name.split('.')
            for attr in attr_chain[:-1]:
                parent = getattr(parent, attr)
            setattr(parent, attr_chain[-1], FCWrapper(module, alpha, r))
    return model