import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class DetectionBranch(nn.Module):
    def __init__(self):
        super(DetectionBranch,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1,stride=1,padding=0,bias=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1

        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,t=1):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)
        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        
        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        return d2


class NonMaxSuppression(nn.Module):
    def __init__(self, n_peaks=256):
        super(NonMaxSuppression,self).__init__()
        self.k = 3 # kernel
        self.p = 1 # padding
        self.s = 1 # stride
        self.center_idx = self.k**2//2
        self.sigmoid = nn.Sigmoid()
        self.unfold = nn.Unfold(kernel_size=self.k, padding=self.p, stride=self.s)
        self.n_peaks = n_peaks

    def sample_peaks(self, x):
        B, _, H, W = x.shape
        for b in range(B):
            x_b = x[b,0]
            idx = torch.topk(x_b.flatten(), self.n_peaks).indices
            idx_i = torch.div(idx, W, rounding_mode='floor')
            idx_j = idx % W
            idx = torch.cat((idx_i.unsqueeze(1), idx_j.unsqueeze(1)), dim=1)
            idx = idx.unsqueeze(0)

            if b == 0:
                graph = idx
            else:
                graph = torch.cat((graph, idx), dim=0)

        return graph 

    def forward(self, feat):
        B, C, H, W = feat.shape

        x = self.sigmoid(feat)

        # Prepare filter
        f = self.unfold(x).view(B, self.k**2, H, W)
        f = torch.argmax(f, dim=1).unsqueeze(1)
        f = (f == self.center_idx).float()

        # Apply filter
        x = x * f

        # Sample top peaks
        graph = self.sample_peaks(x)
        return x, graph

