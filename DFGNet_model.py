import torch
import torch.nn as nn
import torch.nn.functional as F

class DFGNet(nn.Module):
    def __init__(self, nChannel, nDenselayer, nFeat):
        super(DFGNet, self).__init__()
        self.nChannel = nChannel
        self.nDenselayer = nDenselayer
        self.nFeat = nFeat

        self.conv1_1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1)
        self.prelu1_1 = nn.PReLU(nFeat)
        self.conv1_2 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1)
        self.prelu1_2 = nn.PReLU(nFeat)
        self.conv1_3 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1)
        self.prelu1_3 = nn.PReLU(nFeat)

        self.conv2_1 = nn.Conv2d(nFeat, nFeat*2, kernel_size=3, padding=1, stride=2)
        self.prelu2_1 = nn.PReLU(nFeat*2)
        self.conv2_2 = nn.Conv2d(nFeat, nFeat*2, kernel_size=3, padding=1, stride=2)
        self.prelu2_2 = nn.PReLU(nFeat*2)
        self.conv2_3 = nn.Conv2d(nFeat, nFeat*2, kernel_size=3, padding=1, stride=2)
        self.prelu2_3 = nn.PReLU(nFeat*2)

        self.conv3_1 = nn.Conv2d(nFeat*2, nFeat*4, kernel_size=3, padding=1, stride=2)
        self.prelu3_1 = nn.PReLU(nFeat*4)
        self.conv3_2 = nn.Conv2d(nFeat*2, nFeat*4, kernel_size=3, padding=1, stride=2)
        self.prelu3_2 = nn.PReLU(nFeat*4)
        self.conv3_3 = nn.Conv2d(nFeat*2, nFeat*4, kernel_size=3, padding=1, stride=2)
        self.prelu3_3 = nn.PReLU(nFeat*4)


        self.skff1 = SKFF_EN_FC(nFeat,3)
        self.skff2 = SKFF_EN_FC(nFeat*2,3)
        self.skff3 = SKFF_EN_FC(nFeat*4,3)
        #self.skff4 = SKFF_EN(nFeat*8,3)
        self.sksp1 = SKFF_SP(nFeat)
        self.sksp2 = SKFF_SP(nFeat*2)
        self.sksp3 = SKFF_SP(nFeat*4)

        self.up = nn.Upsample(scale_factor=2)
        self.relu = nn.LeakyReLU()
        self.tconv1 = nn.ConvTranspose2d(nFeat*8, nFeat*2, 4, 2, 1)
        self.fuse_3_prelu = nn.PReLU(nFeat*2)
        self.tconv2 = nn.ConvTranspose2d(nFeat*4+nFeat*2, nFeat, 4, 2, 1)
        self.fuse_2_prelu = nn.PReLU(nFeat)
        #self.tconv1 = nn.ConvTranspose2d(n_feat*2, n_feat, 4, 2, 1)
        #self.in_conv = nn.Conv2d(n_feat)

        #self.conv_tail = nn.Conv2d(nFeat*3, nFeat, kernel_size=3, padding=1, bias=True)
        self.conv_tail = nn.Conv2d(nFeat*3,nFeat, kernel_size=3, padding=2, dilation=2)
        self.tail_prelu = nn.PReLU(nFeat)
        self.conv_tail2 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)

    def forward(self, x1, x2, x3, lumi_feats, spatial_feats_1, spatial_feats_2, spatial_feats_3):
        out1_1 = self.conv1_1(x1)
        out2_1 = self.conv1_2(x2)
        out3_1 = self.conv1_3(x3)
 
        out1_1 = self.prelu1_1(out1_1)
        out2_1 = self.prelu1_2(out2_1)
        out3_1 = self.prelu1_3(out3_1)

        out1_2 = self.conv2_1(out1_1)
        out2_2 = self.conv2_2(out2_1)
        out3_2 = self.conv2_3(out3_1)

        out1_2 = self.prelu2_1(out1_2)
        out2_2 = self.prelu2_2(out2_2)
        out3_2 = self.prelu2_3(out3_2)

        out1_3 = self.conv3_1(out1_2)
        out2_3 = self.conv3_2(out2_2)
        out3_3 = self.conv3_3(out3_2)

        out1_3 = self.prelu3_1(out1_3)
        out2_3 = self.prelu3_2(out2_3)
        out3_3 = self.prelu3_3(out3_3)
        
        sk_3 = self.skff3([out1_3, out2_3, out3_3],lumi_feats) + out2_3
        sp_3 = self.sksp3([out1_3, out2_3, out3_3],spatial_feats_3) + out2_3
        fuse_3 = torch.cat([sk_3, sp_3], dim=1)
        fuse_3 = self.tconv1(fuse_3)

        fuse_3 = self.fuse_3_prelu(fuse_3)

        sk_2 = self.skff2([out1_2, out2_2, out3_2],lumi_feats) + out2_2
        sp_2 = self.sksp2([out1_2, out2_2, out3_2],spatial_feats_2) + out2_2
        fuse_2 = torch.cat([sk_2, sp_2], dim=1)
        fuse_2 = torch.cat([fuse_2,fuse_3],dim=1)
        fuse_2 = self.tconv2(fuse_2)

        fuse_2 = self.fuse_2_prelu(fuse_2)

        sk_1 = self.skff1([out1_1, out2_1, out3_1],lumi_feats) + out2_1
        sp_1 = self.sksp1([out1_1, out2_1, out3_1],spatial_feats_1) + out2_1
        fuse_1 = torch.cat([sk_1, sp_1], dim=1)
        fuse_1 = torch.cat([fuse_2, fuse_1], dim=1)

        out = self.conv_tail(fuse_1)

        out = self.tail_prelu(out)
        out = self.conv_tail2(out)
        
        output = nn.functional.tanh(out)

        return output
