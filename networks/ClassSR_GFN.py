
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
from networks.GFN import Net


class ClassSR(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(ClassSR, self).__init__()
        self.upscale=4
        self.classifier=Classifier()
        self.net1 = Net(64, 3)
        self.net2 = Net(52, 2)
        self.net3 = Net(36, 1)

    def forward(self, x, is_train):
        if is_train:
            for i in range(len(x)):
                # print(x[i].unsqueeze(0).shape)
                type = self.classifier(x[i].unsqueeze(0))
                p = F.softmax(type, dim=1)

                p1 = p[0][0]
                p2 = p[0][1]
                p3 = p[0][2]

                deblur_out1, out1 = self.net1(x[i].unsqueeze(0), True, not is_train)
                deblur_out2, out2 = self.net2(x[i].unsqueeze(0), True, not is_train)
                deblur_out3, out3 = self.net3(x[i].unsqueeze(0), True, not is_train)


                out = out1 * p1 + out2 * p2 + out3 * p3

                if i == 0:
                    out_res = out
                    type_res = p
                else:
                    out_res = torch.cat((out_res, out), 0)
                    type_res = torch.cat((type_res, p), 0)
        else:

            for i in range(len(x)):
                type = self.classifier(x[i].unsqueeze(0))

                flag = torch.max(type, 1)[1].data.squeeze()
                p = F.softmax(type, dim=1)
                #flag=np.random.randint(0,2)
                #flag=2
                if flag == 0:
                    deblur_out, out = self.net1(x[i].unsqueeze(0))
                elif flag==1:
                    deblur_out, out = self.net2(x[i].unsqueeze(0))
                elif flag==2:
                    deblur_out, out = self.net3(x[i].unsqueeze(0))
                if i == 0:
                    out_res = out
                    type_res = p
                else:
                    out_res = torch.cat((out_res, out), 0)
                    type_res = torch.cat((type_res, p), 0)

            return out_res, type_res

        return out_res,type_res

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
    ceshi  = torch.rand(2,3,64,64)

    net = Classifier()
    out = net(ceshi)
    p = F.softmax(out, dim=1)
    flag = torch.max(p, 1)[1]
    print(p)
    print(flag)

