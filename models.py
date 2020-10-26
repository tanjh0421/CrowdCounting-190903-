import torch
import torch.nn as nn
import torchvision.models as models


class Effcient(nn.Module):
    def __init__(self, pr=None):
        super(Effcient, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.back_end = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU()
        )

        self._initialize_weights()
        if pr is not None:
            print("\033[1;35mLoad >>> pretrained model ... \033[0m")
            if torch.cuda.is_available():
                pretrained = torch.load(pr).state_dict()
            else:
                pretrained = torch.load(pr, map_location='cpu').state_dict()
            self.load_state_dict(pretrained)
            print("\033[1;35mFinish <<< load pretrained model! \033[0m")
        else:
            print('\033[1;35mLoad pretrained model-VGG16......\033[0m')
            vgg16 = models.vgg16(pretrained=True)
            model_dict = self.state_dict()
            pretrained_dict = vgg16.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print('\033[1;35mFinish load pretrained model-VGG16!\033[0m')

    def forward(self, x):
        x = (x / 255)
        x[:, 0, :, :] = (x[:, 0, :, :] - 0.485) / 0.229
        x[:, 1, :, :] = (x[:, 1, :, :] - 0.456) / 0.224
        x[:, 2, :, :] = (x[:, 2, :, :] - 0.406) / 0.225
        x = self.features(x)
        x = self.back_end(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    md = Effcient()
    a = md(torch.ones(1, 3, 64, 64))
    print(a.shape)