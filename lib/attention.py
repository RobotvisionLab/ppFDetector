import torch
import torch.nn as nn
import torchvision
from functools import reduce

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)             # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


if __name__ == '__main__':

    class Test_Attention(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(Test_Attention, self).__init__()

            self.conv_1 = nn.Conv2d(in_ch, 16, 3, padding=1)
            self.senet = SE_Block(ch_in = 16)
            self.conv_2 = nn.Conv2d(16, 16, 3, padding=1)
            self.cbam = CBAM(channel = 16)
            self.conv_3 = nn.Conv2d(16, out_ch, 1, padding=0)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.conv_1(x)
            print('--> conv1')
            x = self.relu(x)
            x = self.senet(x)
            print('--> senet')
            x = self.conv_2(x)
            print('--> conv2')
            x = self.relu(x)
            x = self.cbam(x)
            print('--> cbam')
            x = self.conv_3(x)
            print('--> conv3')
            print('--> done.')
            return x

    x = torch.randn(1, 1, 16, 16)
    test_net = Test_Attention(1, 1)
    y = test_net(x)
    print('==========+> output: ', y.shape)
    print(test_net)





