import torch
import torch.nn as nn
class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )
    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc=12):
        super(Discriminator, self).__init__()
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = nn.Sequential(
                nn.Conv2d(nc, nf, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                )
 
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        self.clf = nn.Conv2d(nf * 8, 1, 4, 1, 0)
        self.nlin = nn.Sigmoid()

    def forward(self, input):
        h1 = self.c1(input)
        #print(input.size())
        #print(h1.size())
        h2 = self.c2(h1)
        #print(h2.size())
        h3 = self.c3(h2)
        #print(h3.size())
        h4 = self.c4(h3)
        #print(h4.size())
        h5 = self.clf(h4)
        #print(h5.size())
        return h5.view(h5.size()[0])
        return self.nlin(h5).view(h5.size()[0])



