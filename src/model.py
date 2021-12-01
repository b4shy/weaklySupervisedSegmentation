
import torch
import torch.nn as nn
from torch.nn.modules import conv
import torchvision
import torch.nn.functional as F
from petws.src.utils import load_weights

class VGG16PetClassification(torch.nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg.features[0] =  torch.nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg.classifier[6] = torch.nn.Linear(in_features=4096, out_features=1, bias=True)
        
    def forward(self, x):
        return self.vgg(x)
        
class VGG16PetClassification32(torch.nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        del self.vgg.features[4]  # del first max pool
        self.vgg.features[0] =  torch.nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg.classifier[6] = torch.nn.Linear(in_features=4096, out_features=1, bias=True)
        
    def forward(self, x):
        return self.vgg(x)


class VGG16PetClassificationWith1x1(torch.nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        model = torchvision.models.vgg16(pretrained=True)
        model.features[0] =  torch.nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[0] = torch.nn.Linear(in_features=64, out_features=4096, bias=True)
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=1, bias=True)
        self.classifier = model.classifier
        self.features = model.features
        self.last_conv = torch.nn.Conv2d(512, 1, 1)
        

    def forward(self, x):
        features = self.features(x)
        last_conv = self.last_conv(features)
        x = last_conv.view(1, -1)        
        return self.classifier(x)


class VGG16PetClassification32UpConv(torch.nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        model = torchvision.models.vgg16(pretrained=True)
        model.features[0] =  torch.nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=1, bias=True)
        
        self.feature_conv = torch.nn.Sequential(*list(model.features)[:30])
        self.up_conv = torch.nn.ConvTranspose2d(512, 512, 2, 2)
        self.dense = torch.nn.Sequential(*list(model.classifier))

    def forward(self, x):
        x = self.feature_conv(x)
        x = self.up_conv(x)
        features = F.relu(x)
        x = F.max_pool2d(features, 2, 2)
        x = F.adaptive_avg_pool2d(x, output_size=(7, 7))
        x = torch.flatten(x, 1)
        return self.dense(x)


class ResNetClassification(torch.nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)
        self.resnet.conv1 =  torch.nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.resnet.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
        
    def forward(self, x):
        return self.resnet(x)


class ResNetClassification16(torch.nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)
        self.resnet.conv1 =  torch.nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.resnet.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
        
    def forward(self, x):
        return self.resnet(x)

class ResNetClassification32(torch.nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)
        self.resnet.conv1 =  torch.nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.resnet.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
        self.resnet.layer4[0].conv1 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.resnet.layer4[0].downsample[0] = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x):
        return self.resnet(x)


class VggScore32(torch.nn.Module):
    def __init__(self, path="../outputs/Train_VGG_PETCT/07-11-2020_19:48:26/22.ckt"):
        super().__init__()
        model = torchvision.models.vgg16(pretrained=True)
        del model.features[4]
        model.features[0] =  torch.nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=1, bias=True)

        new_state = load_weights(path, 4)
        model.load_state_dict(new_state)

        self.feature_conv = torch.nn.Sequential(*list(model.features)[:29])
        self.dense = torch.nn.Sequential(*list(model.classifier))
            
    def forward(self, x):
        features = self.feature_conv(x)
        
        x = F.max_pool2d(features, 2, 2)
        x = F.adaptive_avg_pool2d(x, output_size=(7, 7))
        x = x.view(1, -1)
        return self.dense(x), features

class VggGrad32(torch.nn.Module):
    def __init__(self, path="../outputs/Train_VGG_PETCT/07-11-2020_19:48:26/22.ckt"):
        super().__init__()
        model = torchvision.models.vgg16(pretrained=True)
        del model.features[4]
        model.features[0] =  torch.nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=1, bias=True)

        new_state = load_weights(path, 4)
        model.load_state_dict(new_state)

        self.feature_conv = torch.nn.Sequential(*list(model.features)[:29])
        self.dense = torch.nn.Sequential(*list(model.classifier))
        
    def activation_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        features = self.feature_conv(x)
        h = features.register_hook(self.activation_hook)
        
        x = F.max_pool2d(features, 2, 2)
        x = F.adaptive_avg_pool2d(x, output_size=(7, 7))
        x = x.view(1, -1)
        return self.dense(x), features
    
    def get_activations_gradient(self):
        return self.gradients


class VGG16PetClassification32UpConvScore(torch.nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        model = torchvision.models.vgg16(pretrained=True)
        model.features[0] =  torch.nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=1, bias=True)
        
        
        self.feature_conv = torch.nn.Sequential(*list(model.features)[:30])
        self.up_conv = torch.nn.ConvTranspose2d(512, 512, 2, 2)
        self.dense = torch.nn.Sequential(*list(model.classifier))
        

    def forward(self, x):
        x = self.feature_conv(x)
        x = self.up_conv(x)
        features = F.relu(x)
        x = F.max_pool2d(features, 2, 2)
        x = F.adaptive_avg_pool2d(x, output_size=(7, 7))
        x = torch.flatten(x, 1)
        return self.dense(x), features

class VGG16PetClassification32UpConvGrad(torch.nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        model = torchvision.models.vgg16(pretrained=True)
        model.features[0] =  torch.nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=1, bias=True)
        
        
        self.feature_conv = torch.nn.Sequential(*list(model.features)[:30])
        self.up_conv = torch.nn.ConvTranspose2d(512, 512, 2, 2)
        self.dense = torch.nn.Sequential(*list(model.classifier))

    def activation_hook(self, grad):
        self.gradients = grad


    def forward(self, x):
        x = self.feature_conv(x)
        x = self.up_conv(x)
        features = F.relu(x)
        h = features.register_hook(self.activation_hook)

        x = F.max_pool2d(features, 2, 2)
        x = F.adaptive_avg_pool2d(x, output_size=(7, 7))
        x = torch.flatten(x, 1)
        return self.dense(x), features

    def get_activations_gradient(self):
        return self.gradients


        
class ResNetGrad32(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        model = torchvision.models.resnet34()
        model.conv1 =  torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
        model.layer4[0].conv1 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.layer4[0].downsample[0] = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

        new_state = load_weights(path, 7)
        model.load_state_dict(new_state)
        
        self.feature_conv = torch.nn.Sequential(*list(model.children())[:-2])
        self.fc = model.fc

    def activation_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        features = self.feature_conv(x)
        h = features.register_hook(self.activation_hook)
        x = F.adaptive_avg_pool2d(features, output_size=(1, 1))

        x = x.view(1, -1)
        return self.fc(x), features

    def get_activations_gradient(self):
        return self.gradients


class ResNetScore32(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        model = torchvision.models.resnet34()
        model.conv1 =  torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
        model.layer4[0].conv1 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.layer4[0].downsample[0] = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        
        new_state = load_weights(path, 7)
        model.load_state_dict(new_state)
        
        self.feature_conv = torch.nn.Sequential(*list(model.children())[:-2])
        self.fc = model.fc


    def forward(self, x):
        features = self.feature_conv(x)
        x = F.adaptive_avg_pool2d(features, output_size=(1, 1))

        x = x.view(1, -1)
        return self.fc(x), features


class BaseNet(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.max_pool = torch.nn.MaxPool2d(2, 2)
        
        self.conv3 = torch.nn.Conv2d(32+2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(98, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = torch.relu(conv1)

        conv2 = self.conv2(conv1)
        conv2 = torch.relu(conv2)
        skip = torch.cat([x, conv2], dim=1)

        max_pooled = self.max_pool(skip)

        conv3 = self.conv3(max_pooled)
        conv3 = torch.relu(conv3)

        conv4 = self.conv4(conv3)
        conv4 = torch.relu(conv4)
        skip2 = torch.cat([max_pooled, conv4], dim=1)
        features = skip2

        gap_pooled = self.gap(features)
        gap_pooled = torch.flatten(gap_pooled, 1)

        output = self.fc(gap_pooled)
        return output


class GPUnet(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        
        self.conv3 = torch.nn.Conv2d(64+2, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv5 = torch.nn.Conv2d(128+66, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        
        self.conv7 = torch.nn.Conv2d(256+128+194, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv8 = torch.nn.Conv2d(128+64+2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, 1)
    
    
    def forward(self, x):
        conv1 = torch.relu(self.conv1(x))

        conv2 = torch.relu(self.conv2(conv1))
        
        skip1 = torch.cat([conv2, x], dim=1)

        max_pooled = self.max_pool(skip1)

        conv3 = torch.relu(self.conv3(max_pooled))

        conv4 = torch.relu(self.conv4(conv3))
        
        skip2 = torch.cat([conv4, max_pooled], dim=1)

        max_pooled2 = self.max_pool(skip2)

        conv5 = torch.relu(self.conv5(max_pooled2))
        conv6 = torch.relu(self.conv6(conv5))

        skip3 = torch.cat([conv6, max_pooled2], dim=1)
    
    
        upsampled = self.upsample(skip3)
        skip4 = torch.cat([upsampled, conv4], dim=1)
        conv7 = self.conv7(skip4)
        
        upsampled2 = self.upsample(conv7)
        skip5 = torch.cat([upsampled2, conv2, x], dim=1)
        conv8 = self.conv8(skip5)

        features = conv8
        
        gap_pooled = self.gap(features)
        gap_pooled = torch.flatten(gap_pooled, 1)

        output = self.fc(gap_pooled)
        return output

class GPUnetScore(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        
        self.conv3 = torch.nn.Conv2d(64+2, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv5 = torch.nn.Conv2d(128+66, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        
        self.conv7 = torch.nn.Conv2d(256+128+194, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv8 = torch.nn.Conv2d(128+64+2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, 1)
    
    
    def forward(self, x):
        conv1 = torch.relu(self.conv1(x))

        conv2 = torch.relu(self.conv2(conv1))
        
        skip1 = torch.cat([conv2, x], dim=1)

        max_pooled = self.max_pool(skip1)

        conv3 = torch.relu(self.conv3(max_pooled))

        conv4 = torch.relu(self.conv4(conv3))
        
        skip2 = torch.cat([conv4, max_pooled], dim=1)

        max_pooled2 = self.max_pool(skip2)

        conv5 = torch.relu(self.conv5(max_pooled2))
        conv6 = torch.relu(self.conv6(conv5))

        skip3 = torch.cat([conv6, max_pooled2], dim=1)
    
    
        upsampled = self.upsample(skip3)
        skip4 = torch.cat([upsampled, conv4], dim=1)
        conv7 = self.conv7(skip4)
        
        upsampled2 = self.upsample(conv7)
        skip5 = torch.cat([upsampled2, conv2, x], dim=1)
        conv8 = self.conv8(skip5)

        features = conv8
        
        gap_pooled = self.gap(features)
        gap_pooled = torch.flatten(gap_pooled, 1)

        output = self.fc(gap_pooled)
        return output, features


class GPUnetGrad(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        
        self.conv3 = torch.nn.Conv2d(64+2, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv5 = torch.nn.Conv2d(128+66, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        
        self.conv7 = torch.nn.Conv2d(256+128+194, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv8 = torch.nn.Conv2d(128+64+2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, 1)
    
    def activation_hook(self, grad):
        self.gradients = grad
    

    def forward(self, x):
        conv1 = torch.relu(self.conv1(x))

        conv2 = torch.relu(self.conv2(conv1))
        
        skip1 = torch.cat([conv2, x], dim=1)

        max_pooled = self.max_pool(skip1)

        conv3 = torch.relu(self.conv3(max_pooled))

        conv4 = torch.relu(self.conv4(conv3))
        
        skip2 = torch.cat([conv4, max_pooled], dim=1)

        max_pooled2 = self.max_pool(skip2)

        conv5 = torch.relu(self.conv5(max_pooled2))
        conv6 = torch.relu(self.conv6(conv5))

        skip3 = torch.cat([conv6, max_pooled2], dim=1)
    
    
        upsampled = self.upsample(skip3)
        skip4 = torch.cat([upsampled, conv4], dim=1)
        conv7 = self.conv7(skip4)
        
        upsampled2 = self.upsample(conv7)
        skip5 = torch.cat([upsampled2, conv2, x], dim=1)
        conv8 = self.conv8(skip5)

        features = conv8
        h = features.register_hook(self.activation_hook)

        gap_pooled = self.gap(features)
        gap_pooled = torch.flatten(gap_pooled, 1)

        output = self.fc(gap_pooled)
        return output, features

    def get_activations_gradient(self):
        return self.gradients
