
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
from networks.GFN import Net
from torchstat import stat

class ClassSR(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(ClassSR, self).__init__()
        self.upscale=4
        self.classifier=Classifier()
        self.net1 = Net(64, 3)
        self.net2 = Net(52, 2)
        self.net3 = Net(36, 1)

    def forward(self, x):
        for i in range(len(x)):
            type = self.classifier(x[i].unsqueeze(0))

            flag = torch.max(type, 1)[1].data.squeeze()
            # p = F.softmax(type, dim=1)
                #flag=np.random.randint(0,2)
                #flag=2
            if flag == 0:
                deblur_out, out = self.net1(x[i].unsqueeze(0))
            elif flag==1:
                deblur_out, out = self.net2(x[i].unsqueeze(0))
            elif flag==2:
                deblur_out, out = self.net3(x[i].unsqueeze(0))
            if i == 0:
                deblur_res = deblur_out
                out_res = out
                type_res = type
            else:
                deblur_res = torch.cat((deblur_res,deblur_out),0)
                out_res = torch.cat((out_res, out), 0)
                type_res = torch.cat((type_res, type), 0)

        return deblur_res, out_res, type_res


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lastOut = nn.Linear(32, 3)

        # Condtion network
        self.CondNet = nn.Sequential(nn.Conv2d(3, 128, 4, 4), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 32, 1))

        for m in self.CondNet.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= 0.1  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AvgPool2d(out.size()[2])(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out)
        return out


if __name__ == '__main__':

    net = ClassSR()

    stat(net.classifier, (3, 64, 64))

