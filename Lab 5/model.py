import torch
import torch.nn as nn

#Basic Block
class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, batch=True):
        super(ResidualBlock, self).__init__()
        if batch:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias= False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, bias= False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias= False),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
            )

        self.skip = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1:
            if batch:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch)
                )
            else:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                )

    def forward(self, x):
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out += self.skip(x)
        x_out = self.relu(x_out)
        return x_out



#Sub-Group
class ResidualLayer(nn.Module):

    def __init__(self, in_ch, out_ch, num_blocks, stride, batch=True):
        super(ResidualLayer, self).__init__()
        blocks = []
        blocks.append(ResidualBlock(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, batch=batch))
        for _ in range(num_blocks-1):
            block = ResidualBlock(out_ch, out_ch, kernel_size=3, stride=1, padding=1, batch=batch)
            blocks.append(block)

        self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        x_out = self.layer(x)
        return x_out
        
#Model
class Resnet18(nn.Module):

    def __init__(self, num_blocks:list, strides:list, batch=True):
        super(Resnet18, self).__init__()
        if batch:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU()
            )

        self.layer1 = ResidualLayer(64, 64, num_blocks[0], strides[0], batch)
        self.layer2 = ResidualLayer(64, 128, num_blocks[1], strides[1], batch)
        self.layer3 = ResidualLayer(128, 256, num_blocks[2], strides[2], batch)
        self.layer4 = ResidualLayer(256, 512, num_blocks[3], strides[3], batch)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x_out = self.conv1(x)
        x_out = self.layer1(x_out)
        x_out = self.layer2(x_out)
        x_out = self.layer3(x_out)
        x_out = self.layer4(x_out)
        x_out = self.avg_pool(x_out)
        x_out = x_out.view(x_out.size(0), -1)
        x_out = self.fc(x_out)

        return x_out


# model = Resnet18([2,2,2,2], [1,2,2,2], False)
# print(model)
# print(type(model))

