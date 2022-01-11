import torch
from torch import nn
from lib.FJDB import FJDB
from lib.feature import Feature
from lib.se_nets import SEBasicBlock


class FJDN(nn.Module):
    def __init__(self):
        super(FJDN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.up_1 = FJDB(block_num=4, inter_channel=32, channel=64)
        self.up_2 = FJDB(block_num=4, inter_channel=32, channel=64)
        self.up_3 = FJDB(block_num=4, inter_channel=32, channel=64)

        self.down_3 = FJDB(block_num=4, inter_channel=32, channel=64)
        self.down_2 = FJDB(block_num=4, inter_channel=32, channel=64)
        self.down_1 = FJDB(block_num=4, inter_channel=32, channel=64)

        self.down_2_fusion = nn.Conv2d(64 + 64, 64, 1, 1, 0)
        self.down_1_fusion = nn.Conv2d(64 + 64, 64, 1, 1, 0)

        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 3, 64, 1, 1, 0),
            nn.Conv2d(64, 64, 3, 1, 1),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh(),
        )
        self.feature = Feature(in_channel=3)

        # self.se_attention = SEBasicBlock(inplanes=64, planes=64)
        # self.se_attention = SEBasicBlock(inplanes=3, planes=3)

    def forward(self, x):
        feature_neg_1 = self.feature(x)
        # print('feature_neg_1:', feature_neg_1.shape)      # [1, 64, 240, 256]
        feature_0 = self.conv2(feature_neg_1)

        up_1_banch = self.up_1(feature_0)
        up_1, indices_1 = nn.MaxPool2d(2, 2, return_indices=True)(up_1_banch)

        up_2 = self.up_2(up_1)
        up_2, indices_2 = nn.MaxPool2d(2, 2, return_indices=True)(up_2)

        up_3 = self.up_3(up_2)
        up_3, indices_3 = nn.MaxPool2d(2, 2, return_indices=True)(up_3)

        down_3 = self.down_3(up_3)

        down_3 = nn.MaxUnpool2d(2, 2)(down_3, indices_3, output_size=up_2.size())

        down_3 = torch.cat([up_2, down_3], dim=1)
        down_3 = self.down_2_fusion(down_3)
        down_2 = self.down_2(down_3)

        down_2 = nn.MaxUnpool2d(2, 2)(down_2, indices_2, output_size=up_1.size())

        down_2 = torch.cat([up_1, down_2], dim=1)
        down_2 = self.down_1_fusion(down_2)
        down_1 = self.down_1(down_2)
        down_1 = nn.MaxUnpool2d(2, 2)(down_1, indices_1, output_size=feature_0.size())

        down_1 = torch.cat([feature_0, down_1], dim=1)

        cat_block_feature = torch.cat([down_1, up_1_banch], 1)
        feature = self.fusion(cat_block_feature)
        # print('feature :', feature.shape)     # [1, 64, 240, 256]
        # feature = feature * self.se_attention(feature_neg_1)
        # feature = feature + feature_neg_1
        #
        # outputs = self.final_conv(feature)

        feature = self.final_conv(feature)
        # feature = feature * self.se_attention(x)

        outputs = x - feature

        return outputs
