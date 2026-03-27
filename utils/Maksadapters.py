import torch
import torch.nn as nn
class thresholdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0.0).float()
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out
class BaseMaskedFC(nn.Module):
    def __init__(self, fc: nn.Linear, num_bitrates=6):
        super().__init__()
        self.fc = fc
        self.num_bitrates = num_bitrates
        self.bitrate_idx = 0
        self.mask = nn.Parameter(torch.ones(num_bitrates, fc.out_features))

    def set_bitrate(self, idx: int):
        self.bitrate_idx = int(idx)

    def forward(self, x):
        base = self.fc(x)
        g = thresholdFunction.apply(self.mask[self.bitrate_idx])
        g = g.view(1, 1, -1)
        return base * g
class BaseMaskedWithAdapterFC(BaseMaskedFC):
    def __init__(self, fc: nn.Linear, num_bitrates=6, alpha=8.0, r=8):
        super().__init__(fc, num_bitrates)

        self.wa = nn.Linear(fc.in_features, r)
        self.wb = nn.Linear(r, fc.out_features)

        nn.init.zeros_(self.wa.weight)
        nn.init.zeros_(self.wa.bias)
        nn.init.normal_(self.wb.weight, 0.0, 1.0)
        nn.init.zeros_(self.wb.bias)

        self.scale = alpha / r

    def forward(self, x):
        pruned_base = super().forward(x)
        adapter = self.scale * self.wb(self.wa(x))
        return pruned_base + adapter
    
def apply_mask_to_all_linear(model, num_bitrates=6):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, BaseMaskedFC(module, num_bitrates))
        else:
            apply_mask_to_all_linear(module, num_bitrates)
    return model
def add_adapters_to_mlp(model, alpha=8, r=8, num_bitrates=6):
    for name, module in model.named_modules():
        if type(module).__name__ == 'Mlp':
            module.fc1 = BaseMaskedWithAdapterFC(
                module.fc1.fc if isinstance(module.fc1, BaseMaskedFC) else module.fc1,
                num_bitrates, alpha, r
            )
            module.fc2 = BaseMaskedWithAdapterFC(
                module.fc2.fc if isinstance(module.fc2, BaseMaskedFC) else module.fc2,
                num_bitrates, alpha, r
            )
    return model