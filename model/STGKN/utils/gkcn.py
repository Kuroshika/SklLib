import torch
import torch.nn as nn
from model.STGKN.utils.KANConv import KAN_Convolution,KAN_Convolutional_Layer


class unit_gkcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 adaptive='importance',
                 conv_pos='pre',
                 with_res=False,
                 norm='BN',
                 act='ReLU',
                 n_convs=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)

        assert adaptive in [None, 'init', 'offset', 'importance']
        self.adaptive = adaptive
        assert conv_pos in ['pre', 'post']
        self.conv_pos = conv_pos
        self.with_res = with_res

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        # self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        # self.bn = nn.BatchNorm2d(10)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.act = build_activation_layer(self.act_cfg)
        self.act = nn.ReLU()

        if self.adaptive == 'init':
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)

        if self.adaptive in ['offset', 'importance']:
            self.PA = nn.Parameter(A.clone())
            if self.adaptive == 'offset':
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            elif self.adaptive == 'importance':
                nn.init.constant_(self.PA, 1)

        if self.conv_pos == 'pre':
            self.conv = KAN_Convolutional_Layer(
                n_convs=n_convs,
                kernel_size=(1, 1),
                padding=(0, 0),
                device="cuda")
            # self.conv = KAN_Convolution(in_channels, out_channels * A.size(0), 1)
        elif self.conv_pos == 'post':
            self.conv = KAN_Convolution(A.size(0) * in_channels, out_channels, 1)

        if self.with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    KAN_Convolution(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels))
                # build_norm_layer(self.norm_cfg, out_channels)[1])
            else:
                self.down = lambda x: x

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0

        A_switch = {None: self.A, 'init': self.A}
        if hasattr(self, 'PA'):
            A_switch.update({'offset': self.A + self.PA, 'importance': self.A * self.PA})
        A = A_switch[self.adaptive]

        if self.conv_pos == 'pre':
            x = self.conv(x)
            x = x.view(n, self.num_subsets, -1, t, v)
            x = torch.einsum('nkctv,kvw->nctw', (x, A)).contiguous()
        elif self.conv_pos == 'post':
            x = torch.einsum('nctv,kvw->nkctw', (x, A)).contiguous()
            x = x.view(n, -1, t, v)
            x = self.conv(x)
        x = self.bn(x) + res
        return self.act(x)

    def init_weights(self):
        pass
