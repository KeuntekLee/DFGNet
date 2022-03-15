import torch
import torch.nn as nn

class Exposure_Encoder(nn.Module):
    def __init__(self, nFeat, outdim):
        super(Exposure_Encoder, self).__init__()

        #self.stage = stage
        self.nFeat = nFeat
        self.outdim = outdim
        self.refpad_3 = nn.ReflectionPad2d(3)
        self.refpad_1 = nn.ReflectionPad2d(1)
        self.headconv = nn.Conv2d(3,nFeat, kernel_size=7)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(nFeat, nFeat*2, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(nFeat*2, nFeat*4, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(nFeat*4, nFeat*4, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(nFeat*4, nFeat*4, kernel_size=4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(nFeat*4, outdim)
    def forward(self, x):
        x = self.refpad_3(x)
        x = self.headconv(x)
        x = self.relu(x)

        x = self.refpad_1(x)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.refpad_1(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = self.refpad_1(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = self.refpad_1(x)
        x = self.conv4(x)
        x = self.relu(x)

        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

class ResBlock(nn.Module):
    def __init__(self, nFeat):
        super(ResBlock,self).__init__()
        self.relu = nn.ReLU()
        self.ln1 = LayerNorm(nFeat)
        self.ln2 = LayerNorm(nFeat)
        self.refpad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(nFeat, nFeat, kernel_size=3)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3)
    def forward(self,x):
        identity=x
        out = self.ln1(x)
        out = self.relu(out)
        out = self.refpad(out)
        out = self.conv1(out)
        out = self.ln2(out)
        out = self.relu(out)
        out = self.refpad(out)
        out = self.conv2(out)
        return out + identity

class Spatial_Encoder(nn.Module):
    def __init__(self, nFeat):
        super(Spatial_Encoder, self).__init__()
        self.nFeat = nFeat
        #self.outdim = outdim
        self.refpad_3 = nn.ReflectionPad2d(3)
        self.refpad_1 = nn.ReflectionPad2d(1)
        self.headconv = nn.Conv2d(3, nFeat, kernel_size=7)
        self.ln1 = LayerNorm(nFeat)
        self.conv1 = nn.Conv2d(nFeat, nFeat*2, kernel_size=4, stride=2)
        self.ln2 = LayerNorm(nFeat*2)
        self.conv2 = nn.Conv2d(nFeat*2, nFeat*4, kernel_size=4, stride=2)
        self.ln3 = LayerNorm(nFeat*4)
        self.conv3 = nn.Conv2d(nFeat*4, nFeat*8, kernel_size=4, stride=2)

        self.resblock1 = ResBlock(nFeat*8)
        self.resblock2 = ResBlock(nFeat*8)
        self.ln_tail = LayerNorm(nFeat*8)
        self.tailconv = nn.Conv2d(nFeat*8, nFeat*8, kernel_size=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        y = self.refpad_3(x)
        y = self.headconv(y)

        y = self.ln1(y)
        y = self.relu(y)
        y = self.refpad_1(y)
        y = self.conv1(y)

        y = self.ln2(y)
        y = self.relu(y)
        y = self.refpad_1(y)
        y = self.conv2(y)

        y = self.ln3(y)
        y = self.relu(y)
        y = self.refpad_1(y)
        y = self.conv3(y)

        y = self.resblock1(y)

        y = self.resblock2(y)

        y = self.ln_tail(y)
        y = self.relu(y)
        y = self.tailconv(y)

        return y

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y_fc1, y_fc2):
        eps = 1e-5
        mean_x = torch.mean(x, dim=[2,3])
		#mean_y = torch.mean(y, dim=[2,3])

        std_x = torch.std(x, dim=[2,3])
		#std_y = torch.std(y, dim=[2,3])

        mean_x = mean_x.unsqueeze(-1).unsqueeze(-1)
		#mean_y = mean_y.unsqueeze(-1).unsqueeze(-1)

        std_x = std_x.unsqueeze(-1).unsqueeze(-1) + eps
		#std_y = std_y.unsqueeze(-1).unsqueeze(-1) + eps
        y_fc2 = y_fc2.unsqueeze(-1).unsqueeze(-1)
        y_fc1 = y_fc1.unsqueeze(-1).unsqueeze(-1)
        out = (x - mean_x)/ std_x * y_fc2 + y_fc1


        return out

class ResBlockT(nn.Module):
    def __init__(self, nFeat):
        super(ResBlockT,self).__init__()
        self.relu = nn.ReLU()
        self.ln1 = LayerNorm(nFeat)
        self.ln2 = LayerNorm(nFeat)
        #self.refpad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1)
        self.conv2 = nn.ConvTranspose2d(nFeat, nFeat//2, kernel_size=3, padding =1, stride=2, output_padding=1)
        self.conv_identity = nn.ConvTranspose2d(nFeat, nFeat//2, kernel_size=3, padding=1, stride=2, output_padding=1)
    def forward(self,x):
        identity=x
        identity = self.conv_identity(identity)
        out = self.ln1(x)
        out = self.relu(out)
        #out = self.refpad(out)
        out = self.conv1(out)
        out = self.ln2(out)
        out = self.relu(out)
        #out = self.refpad(out)
        out = self.conv2(out)
        return out + identity

class Decoder(nn.Module):
    def __init__(self, nFeat, exposure_vec_dim):
        super(Decoder, self).__init__()
        self.fc_mean = nn.Linear(exposure_vec_dim, nFeat)
        self.fc_var = nn.Linear(exposure_vec_dim, nFeat)
        self.headconv = nn.Conv2d(nFeat, nFeat, kernel_size=1)
        self.adin = AdaIN()
        self.resblock = ResBlock(nFeat)
        self.resblockt1 = ResBlockT(nFeat)
        self.resblockt2 = ResBlockT(nFeat//2)
        self.resblockt3 = ResBlockT(nFeat//4)

        self.ln_tail = LayerNorm(nFeat//8)
        self.relu = nn.ReLU()
        self.refpad = nn.ReflectionPad2d(1)
        self.conv_tail = nn.Conv2d(nFeat//8, 3, kernel_size=3)
    def forward(self, sp, ex):
        mean = self.fc_mean(ex)
        var = self.fc_var(ex)
        affined = self.adin(sp, mean, var)
        out = self.headconv(affined)
        out = self.resblock(out)
        out = self.resblockt1(out)
        out = self.resblockt2(out)
        out = self.resblockt3(out)

        out = self.ln_tail(out)
        out = self.relu(out)
        out = self.refpad(out)
        out = self.conv_tail(out)

        return out
