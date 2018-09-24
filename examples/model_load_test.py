import torch
import torch.nn as nn

#class LoadTest():
class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x


class HardNet(nn.Module):
    def __init__(self):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=2),
            # nn.Conv2d(1, 32, kernel_size=7),
            nn.Tanh(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=6),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(nn.Linear(64 * 8 * 8, 128),
                                        nn.Tanh()
                                        )
        # self.features.apply(weights_init)

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        x = self.classifier(x)
        return L2Norm()(x)

class LoadTest():

    def __init__(self):

        #model=HardNet()
        #model=torch.load('/home/kangrong/HardNet/hardnet/code/data/models/tfeat_whole/liberty_train_tfeat_whole/_liberty_min_as_fliprot/checkpoint_0.pth')
        model=torch.load('/unsullied/sharefs/kangrong/home/hardnet/data/models/model_HPatches_HardTFeat_view_lr01_trimar/all_min_as/checkpoint_8.pth')
        print('ok')
if __name__=="__main__":
    LoadTest()
