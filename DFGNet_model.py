import torch
import torch.nn as nn
import torch.nn.functional as F

class SAU(nn.Module):
    def __init__(self, in_channels, height=3):
        super(SAU, self).__init__()
        
        self.height = height

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.filter1 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.filter2 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.filter3 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.feat_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.LeakyReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats, sp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]

        feat_U = inp_feats[0]+inp_feats[1]+inp_feats[2]
        feat_U = self.relu(self.conv1(feat_U))
        sp1 = self.relu(self.feat_conv(sp_feats[0]))
        sp2 = self.relu(self.feat_conv(sp_feats[1]))
        sp3 = self.relu(self.feat_conv(sp_feats[2]))
        feat_1 = self.filter1(torch.cat([feat_U,sp1],dim=1))
        feat_2 = self.filter2(torch.cat([feat_U,sp2],dim=1))
        feat_3 = self.filter3(torch.cat([feat_U,sp3],dim=1))
        feat_1 = nn.functional.sigmoid(feat_1)
        feat_2 = nn.functional.sigmoid(feat_2)
        feat_3 = nn.functional.sigmoid(feat_3)


        feats_V = inp_feats[0]*feat_1 + inp_feats[1]*feat_2 + inp_feats[2]*feat_3
        
        return feats_V     
    
class EAU(nn.Module):
    def __init__(self, in_channels, height=3):
        super(EAU, self).__init__()
        
        self.height = height
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Linear(1024,in_channels))

        self.softmax = nn.Softmax(dim=1)
        #self.softmax = nn.Sigmoid()

    def forward(self, inp_feats, lumi_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        attention_vectors = [self.fcs[0](lumi_feats[0]), self.fcs[1](lumi_feats[1]), self.fcs[2](lumi_feats[2])]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        #print(attention_vectors.shape)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)

        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V 


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


        self.eau1 = EAU(nFeat,3)
        self.eau2 = EAU(nFeat*2,3)
        self.eau3 = EAU(nFeat*4,3)
        #self.skff4 = SKFF_EN(nFeat*8,3)
        self.sau1 = SAU(nFeat)
        self.sau2 = SAU(nFeat*2)
        self.sau3 = SAU(nFeat*4)

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
        
        sk_3 = self.eau3([out1_3, out2_3, out3_3],lumi_feats) + out2_3
        sp_3 = self.eau3([out1_3, out2_3, out3_3],spatial_feats_3) + out2_3
        fuse_3 = torch.cat([sk_3, sp_3], dim=1)
        fuse_3 = self.tconv1(fuse_3)

        fuse_3 = self.fuse_3_prelu(fuse_3)

        sk_2 = self.sau2([out1_2, out2_2, out3_2],lumi_feats) + out2_2
        sp_2 = self.sau2([out1_2, out2_2, out3_2],spatial_feats_2) + out2_2
        fuse_2 = torch.cat([sk_2, sp_2], dim=1)
        fuse_2 = torch.cat([fuse_2,fuse_3],dim=1)
        fuse_2 = self.tconv2(fuse_2)

        fuse_2 = self.fuse_2_prelu(fuse_2)

        sk_1 = self.eau1([out1_1, out2_1, out3_1],lumi_feats) + out2_1
        sp_1 = self.sau1([out1_1, out2_1, out3_1],spatial_feats_1) + out2_1
        fuse_1 = torch.cat([sk_1, sp_1], dim=1)
        fuse_1 = torch.cat([fuse_2, fuse_1], dim=1)

        out = self.conv_tail(fuse_1)

        out = self.tail_prelu(out)
        out = self.conv_tail2(out)
        
        output = nn.functional.tanh(out)

        return output
