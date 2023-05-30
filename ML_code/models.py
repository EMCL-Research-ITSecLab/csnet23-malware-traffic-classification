import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, num_classes: int, output_channels: list, pool_positions: list, last_pooling=nn.AdaptiveAvgPool2d, dropout: float=0.5, use_fc_layers: bool=False):
        super(VGG, self).__init__()

        self.use_fc_layers = use_fc_layers

        self.conv = nn.Sequential()
        in_channels = 1

        if num_classes == 2:
            self.num_outputs = 1
        else:   
            self.num_outputs = num_classes

        for i, out_channels in enumerate(output_channels):
            self.conv.extend(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False
                        ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                ]
            )
            if i in pool_positions:
                self.conv.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_channels = out_channels
        
        if not self.use_fc_layers:
            self.top = nn.Sequential(
                nn.Conv2d(in_channels=output_channels[-1], out_channels=self.num_outputs, kernel_size=1),
                last_pooling((1, 1))
            )
        
        else:
            self.mid = last_pooling((1,1))

            self.top = nn.Sequential(
                nn.Linear(output_channels[-1], 4096),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(4096, self.num_outputs)
            )

    def forward(self, x):
        # Input batches x width x height   Output batches x 1 width x height
        x = x.view((x.size(0), 1, x.size(1), x.size(2)))
        # Output batches x channel x width x height
        x = self.conv(x)
        if self.use_fc_layers:
            # Ouput batches x channels x 1 x 1
            x = self.mid(x)
            # Output batches x channels
            x = x.view((x.size(0), x.size(1)))
        # Output batches x n_classes if use_fc else batches x n_classes x 1 x 1    
        x = self.top(x)
        if not self.use_fc_layers:
            # Output batches x n_classes
            x = x.view((x.size(0), x.size(1)))

        return x


class ResNextResidualBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, stride=1, identity_downsample = None):
        super(ResNextResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels, kernel_size=3, stride=stride, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels*2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(intermediate_channels*2),
        )
        self.identity_downsample = identity_downsample
        self.last_relu = nn.ReLU()
    
    def forward(self, x):
        identity = x 
        x = self.conv(x)
        if self.identity_downsample:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.last_relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, stride=1, identity_downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels*4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(intermediate_channels*4),
        )
        self.identity_downsample = identity_downsample
        self.last_relu = nn.ReLU()
    
    def forward(self, x):
        identity = x 
        x = self.conv(x)
        if self.identity_downsample:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.last_relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes:int, last_pooling=nn.AdaptiveAvgPool2d, resnext=False):
        super(ResNet, self).__init__()
         
        if num_classes == 2:
            self.num_outputs = 1
        else:
            self.num_outputs = num_classes

        if not resnext:
            self.expansion = 4
            self.residual_block = ResidualBlock
            intermediate_channels = [64, 128, 256, 512]
        else:
            self.expansion = 2
            self.residual_block = ResNextResidualBlock
            intermediate_channels = [128, 256, 512, 1024]


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.create_layer(num_residual_blocks=3, in_channels=64, intermediate_channels=intermediate_channels[0], stride=1)
        self.layer2 = self.create_layer(num_residual_blocks=4, in_channels=intermediate_channels[0]*self.expansion, intermediate_channels=intermediate_channels[1], stride=2)
        self.layer3 = self.create_layer(num_residual_blocks=6, in_channels=intermediate_channels[1]*self.expansion, intermediate_channels=intermediate_channels[2], stride=2)        
        self.layer4 = self.create_layer(num_residual_blocks=3, in_channels=intermediate_channels[2]*self.expansion, intermediate_channels=intermediate_channels[3], stride=2)

        self.pool = last_pooling((1,1))
        self.fc = nn.Linear(intermediate_channels[3]*self.expansion, self.num_outputs)

    def create_layer(self, num_residual_blocks, in_channels, intermediate_channels, stride):
       
        identity_downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(intermediate_channels*self.expansion)
            )
        
        layers = [self.residual_block(in_channels=in_channels, intermediate_channels=intermediate_channels, stride=stride, identity_downsample=identity_downsample)]

        for i in range(num_residual_blocks - 1):
            layers.append(self.residual_block(in_channels=intermediate_channels*self.expansion, intermediate_channels=intermediate_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Input batches x width x height   Output batches x 1 width x height
        x = x.view((x.size(0), 1, x.size(1), x.size(2)))
        # Output batches x channels x width x height
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Ouput batches x channels x 1 x 1
        x = self.pool(x)
        # Output batches x channels
        x = x.view((x.size(0), x.size(1)))
        # Output batches x n_classes
        x = self.fc(x)
        
        return x        
    
