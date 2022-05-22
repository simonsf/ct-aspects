import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict


cam_w = 100
cam_sigma = 0.4


def refine_cams(cam_original, image_shape, using_sigmoid=True):
    cam_original = F.interpolate(
        cam_original, image_shape, mode="trilinear", align_corners=True
    )
    B, C, D, H, W = cam_original.size()
    cams = []
    for idx in range(C):
        cam = cam_original[:, idx, :, :, :]
        cam = cam.view(B, -1)
        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        norm = cam_max - cam_min
        norm[norm == 0] = 1e-5
        cam = (cam - cam_min) / norm
        cam = cam.view(B, D, H, W).unsqueeze(1)
        cams.append(cam)
    cams = torch.cat(cams, dim=1)
    if using_sigmoid:
        cams = torch.sigmoid(cam_w*(cams - cam_sigma))
    return cams


class ConvBnRelu3(nn.Module):
    """ classic combination: conv + batch normalization [+ relu]
        post-activation mode """

    def __init__(self, in_channels, out_channels, ksize, padding, do_act=True, bias=True):
        super(ConvBnRelu3, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=ksize, padding=padding, groups=1, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.do_act = do_act
        if do_act:
            self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.bn(self.conv(input))
        if self.do_act:
            out = self.act(out)
        return out


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias), self.weight

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class WeakenedMAE(nn.Module):
    def __init__(self, baseline=0.5, maximum=0.9):
        super(WeakenedMAE, self).__init__()
        self.baseline = baseline
        self.maximum = maximum

    def forward(self, pred, tag, por):
        base = self.baseline * tag
        add = torch.sqrt(por) / 2.
        target_p = torch.clamp(base + add, max=self.maximum)
        target_n = 0.05 * (1 - tag)
        target = target_p + target_n
        pred_ = pred - tag * torch.clamp((pred - target_p), min=0) + (1 - tag) * torch.clamp((target_n - pred), min=0)
        mse = F.mse_loss(pred_, target)
        print(pred_)
        print(target)
        #mae = torch.mean(L1)
        return mse


class InputBlock(nn.Module):
    """ input block of vb-net """

    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=[1, 3, 3], padding=[0, 1, 1])
        self.bn = nn.BatchNorm3d(out_channels)
        #self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.bn(self.conv(input))
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rate=1, bias=True):
        super(ASPP, self).__init__()
        padding = np.array([0, rate, rate])
        dilate = np.array([rate, rate])
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, [1, 3, 3], 1, padding=list(padding),
                      dilation=[1] + list(dilate), bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, [1, 3, 3], 1, padding=list(2 * padding),
                      dilation=[1] + list(2 * dilate), bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, [1, 3, 3], 1, padding=list(3 * padding),
                      dilation=[1] + list(3 * dilate), bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_cat = nn.Sequential(
            #SELayer(out_channels * 3),
            nn.Conv3d(out_channels * 3, out_channels, 1, 1, padding=0, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        conv3x3_0 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        #conv3x3_3 = self.branch4(x)
        feature_cat = torch.cat([conv3x3_0, conv3x3_1, conv3x3_2], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class ASPP3D(nn.Module):
    def __init__(self, in_channels, out_channels, rate=1, bias=True):
        super(ASPP3D, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, padding=rate, dilation=rate, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, padding=2 * rate, dilation=2 * rate, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, padding=3 * rate, dilation=3 * rate, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_cat = nn.Sequential(
            #SELayer(out_channels * 3),
            nn.Conv3d(out_channels * 3, out_channels, 1, 1, padding=0, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        conv3x3_0 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        #conv3x3_3 = self.branch4(x)
        feature_cat = torch.cat([conv3x3_0, conv3x3_1, conv3x3_2], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class _DenseLayer(nn.Module):
    """Basic unit of DenseBlock (using bottleneck layer) """
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate, aspp=False):
        super(_DenseLayer, self).__init__()
        self.conv1 = ConvBnRelu3(in_channels, bn_size*growth_rate, ksize=1, padding=0, bias=False)
        if not aspp:
            self.conv2 = nn.Conv3d(bn_size*growth_rate, growth_rate,
                                           kernel_size=[1, 3, 3], stride=1, padding=[0, 1, 1], bias=False)
        else:
            self.conv2 = ASPP(bn_size*growth_rate, growth_rate, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.conv2(self.conv1(x))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseLayer3D(nn.Module):
    """Basic unit of DenseBlock (using bottleneck layer) """
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate, aspp=False):
        super(_DenseLayer3D, self).__init__()
        self.conv1 = ConvBnRelu3(in_channels, bn_size*growth_rate, ksize=1, padding=0, bias=False)
        if not aspp:
            self.conv2 = nn.Conv3d(bn_size*growth_rate, growth_rate,
                                           kernel_size=[1, 3, 3], stride=1, padding=[0, 1, 1], bias=False)
            self.conv3 = nn.Conv3d(growth_rate, growth_rate,
                                           kernel_size=[3, 1, 1], stride=1, padding=[1, 0, 0], bias=False)
        else:
            self.conv2 = ASPP3D(bn_size*growth_rate, growth_rate, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.conv3(self.conv2(self.conv1(x)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    """DenseBlock"""
    def __init__(self, num_layers, in_channels, bn_size, growth_rate, drop_rate, dim=2, aspp=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            #if i%2 == 1:
            #    aspp = True
            #else:
            #    aspp = False
            if dim == 2:
                layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size,
                                    drop_rate, aspp=aspp)
            else:
                layer = _DenseLayer3D(in_channels + i * growth_rate, growth_rate, bn_size,
                                    drop_rate, aspp=aspp)
            self.add_module("denselayer%d" % (i+1,), layer)


class _Transition(nn.Module):
    """Transition layer between two adjacent DenseBlock"""
    def __init__(self, num_input_feature, num_output_features, stride_dim=3, down=True):
        super(_Transition, self).__init__()
        if down is True:
            stride = 2
        else:
            stride = 1
        self.conv = ConvBnRelu3(num_input_feature, num_output_features, ksize=1, padding=0, bias=False)
        if stride_dim == 2:
            self.down_conv = nn.Conv3d(num_output_features, num_output_features,
                            kernel_size=[1, 2, 2], stride=[1, stride, stride], bias=False)
            #self.down_pool = nn.MaxPool3d(kernel_size=[1, 2, 2], stride=1)
        elif stride_dim == 3:
            self.down_conv = nn.Conv3d(num_output_features, num_output_features,
                            kernel_size=2, stride=stride, bias=False)
            #self.down_pool = nn.MaxPool3d(kernel_size=2, stride=1)

    def forward(self, x):
        out = self.down_conv(self.conv(x))
        return out


class DenseFeature(nn.Module):
    def __init__(self, in_channels,
                 growth_rate=32,
                 block_config_1=(4, 8),
                 block_config_2=(12, 8),
                 num_init_features=32,
                 bn_size=3,
                 compression_rate=0.5,
                 drop_rate=0.1):
        super(DenseFeature, self).__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv3d(16, 16, kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])),
            ('ASPP0', ASPP(16, num_init_features))
        ]))
        self.features_2 = nn.Sequential(OrderedDict())
        num_features = num_init_features
        for i, num_layers in enumerate(block_config_1):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate=drop_rate, dim=2)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features += num_layers*growth_rate
            if i != len(block_config_1) - 1:
                transition = _Transition(num_features, int(num_features*compression_rate), stride_dim=2)
                self.features.add_module("transition%d" % (i + 1), transition)
                num_features = int(num_features * compression_rate)
                self.features.add_module("z_conv%d" % (i + 1),
                                           nn.Conv3d(num_features,
                                                     num_features,
                                                     kernel_size=[3, 1, 1], padding=[1, 0, 0], bias=False))

        transition = _Transition(num_features, int(num_features * compression_rate), stride_dim=2)
        # self.features_2.add_module("transition%d" % (i + 1), transition)
        self.features.add_module("transition%d" % (i + 1), transition)
        self.features_2.add_module("z_conv%d" % (i + 1),
                                   nn.Conv3d(int(num_features * compression_rate), int(num_features * compression_rate),
                                             kernel_size=[3, 1, 1], padding=[1, 0, 0], bias=False))
        num_features = int(num_features * compression_rate)
        for i, num_layers in enumerate(block_config_2):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate=drop_rate, dim=2)
            self.features_2.add_module("denseblock%d" % (i + 3), block)
            num_features += num_layers * growth_rate
            if i != len(block_config_2) - 1:
                transition = _Transition(num_features, int(num_features * compression_rate), stride_dim=2, down=False)
                self.features_2.add_module("transition%d" % (i + 3), transition)
                num_features = int(num_features * compression_rate)
                self.features_2.add_module("z_conv%d" % (i + 3),
                                           nn.Conv3d(num_features,
                                                     num_features,
                                                     kernel_size=[3, 1, 1], padding=[1, 0, 0], bias=False))
        self.num_features = num_features
        self.features_2.add_module("norm5", nn.BatchNorm3d(num_features))
        self.features_2.add_module("relu5", nn.ReLU(inplace=True))

    def _get_num_feature(self):
        return self.num_features

    def mask_input(self, input):
        if input.shape[1] == 1:
            return input
        img = input[:, [0], :, :, :]
        mask = input[:, [1], :, :, :]
        x = self.in_block(img)
        x = x * mask
        return x

    def forward(self, input):
        #x = self.mask_input(input)
        x = self.in_block(input)
        x = self.features(x)
        x = self.features_2(x)
        return x


class ClassificationNet(nn.Module):
    def __init__(self, in_channels, use_nwu=False, growth_rate=32, block_config_1=(4, 10), block_config_2=(12, 8),
                 num_init_features=32, bn_size=3, compression_rate=0.5, drop_rate=0.1, fc_layer=1, k=None):
        super(ClassificationNet, self).__init__()
        self.feature = DenseFeature(in_channels, growth_rate, block_config_1, block_config_2, num_init_features,
                                    bn_size, compression_rate, drop_rate)

        self.nwu = use_nwu
        assert fc_layer == 1 or fc_layer == 2
        self.fc_layer = fc_layer

        self.gap = nn.AdaptiveAvgPool3d(1)
        num_features = self.feature._get_num_feature()
        layers = [ConvBnRelu3(num_features, int(num_features / 3), ksize=1, padding=0, do_act=True),
                  ConvBnRelu3(int(num_features / 3), 128, ksize=3, padding=1, do_act=False)]
        self.fuse = nn.Sequential(*layers)

        #self.classifier_layer = [nn.Linear(128 * 2, 48), nn.ReLU(), nn.Linear(48, 1)]
        #self.classifier = nn.Sequential(*self.classifier_layer)
        if fc_layer == 1:
            self.classifier = Linear(128 * 2 + int(self.nwu), 1, bias=False)
        else:
            self.classifier1 = Linear(128 * 2 + int(self.nwu), 64, bias=True)
            self.classifier2 = Linear(64, 1, bias=False)

        #self.classifier_layer_bi = [nn.Linear(128 * 3, 64), nn.ReLU(), nn.Linear(64, 1)]
        #self.classifier_bi = nn.Sequential(*self.classifier_layer_bi)
        #self.classifier_bi = Linear(128 * 3, 1, bias=False)

        self.act = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                #nn.init.kaiming_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                try:
                    nn.init.constant_(m.bias, 0)
                except:
                    continue
        self.k = k
        assert self.k is None or self.k > 0
        
    def f1(self, x):
        return 12 * torch.pow((x + 0.05), 3)
    
    def f2(self, x):
        return 1.0275 + 0.056 * torch.log(torch.abs(x - 0.34028))
        
    def cal_alpha(self, x):
        a1 = self.f1(x)
        a2 = self.f2(x)
        alpha = (x < 0.4).float() * a1 + (x >= 0.4).float() * a2
        return alpha.detach()

    def bilatreal_gate(self, y1, y2):
        diff = torch.pow(torch.abs(y1 - y2), self.k)
        #new_y1 = diff * (y1 - diff + 1)
        #new_y2 = diff * (y2 - diff + 1)
        #alpha = torch.pow(torch.abs(y1 - y2), 0.5)
        #alpha = alpha.detach()
        alpha = self.cal_alpha(torch.abs(y1 - y2))
        alpha = alpha.clamp(max = 1)
        new_y1 = alpha * y1 + (1 - alpha) * diff
        new_y2 = alpha * y2 + (1 - alpha) * diff
        return new_y1, new_y2

    def forward(self, input1, input2, nwu_1=None, nwu_2=None):
        if self.nwu is True:
            assert nwu_1 is not None and nwu_2 is not None, 'should input bilateral NWU when use_nwu is set to be True'
        _, _, D, H, W = input1.size()
        bs = input1.shape[0]
        feat1 = self.fuse(self.feature(input1))
        feat2 = self.fuse(self.feature(input2))
        #arr1 = self.gap(feat1).view(bs, -1)
        #arr2 = self.gap(feat2).view(bs, -1)
        sub1 = (feat1 - feat2).float() / (torch.max(abs(feat1), abs(feat2)).float() + 1e-9)
        #sub1 = self.gap(sub1).view(bs, -1)
        sub2 = -1 * sub1
        #sub = abs(sub1)

        arr1 = self.relu(torch.cat([feat1, sub1], dim=1))
        arr2 = self.relu(torch.cat([feat2, sub2], dim=1))
        #arr = self.relu(torch.cat([feat1, sub, feat2], dim=1))

        cl1 = self.gap(arr1).view(bs, -1)
        cl2 = self.gap(arr2).view(bs, -1)
        #cl = self.gap(arr).view(bs, -1)

        if self.nwu is True:
            cl1 = torch.cat([cl1, nwu_1], dim=1)
            cl2 = torch.cat([cl2, nwu_2], dim=1)

        if self.fc_layer == 1:
            out1, fc_l = self.classifier(cl1)
            out2, fc_r = self.classifier(cl2)
            #out, fc_bi = self.classifier_bi(cl)
        else:
            out1_1, fc_l_1 = self.classifier1(cl1)
            out2_1, fc_r_1 = self.classifier1(cl2)
            out1_1 = self.relu(out1_1)
            out2_1 = self.relu(out2_1)
            out1, fc_l_2 = self.classifier2(out1_1)
            out2, fc_r_2 = self.classifier2(out2_1)
            fc_l = torch.matmul(fc_l_2, fc_l_1)
            fc_r = torch.matmul(fc_r_2, fc_r_1)

        if self.nwu is not True:
            cam_l = self.relu(
                F.conv3d(arr1, fc_l.detach().unsqueeze(2).unsqueeze(3).unsqueeze(4), bias=None, stride=1, padding=0))
            cam_l_refined = refine_cams(cam_l, (D, H, W), using_sigmoid=False)

            cam_r = self.relu(
                F.conv3d(arr2, fc_r.detach().unsqueeze(2).unsqueeze(3).unsqueeze(4), bias=None, stride=1, padding=0))
            cam_r_refined = refine_cams(cam_r, (D, H, W), using_sigmoid=False)
        else:
            cam_l = self.relu(
                F.conv3d(arr1, fc_l[:, :-1].detach().unsqueeze(2).unsqueeze(3).unsqueeze(4), bias=None, stride=1, padding=0))
            cam_l_refined = refine_cams(cam_l, (D, H, W), using_sigmoid=False)

            cam_r = self.relu(
                F.conv3d(arr2, fc_r[:, :-1].detach().unsqueeze(2).unsqueeze(3).unsqueeze(4), bias=None, stride=1, padding=0))
            cam_r_refined = refine_cams(cam_r, (D, H, W), using_sigmoid=False)

        #cam = self.relu(
        #    F.conv3d(arr, fc_bi.detach().unsqueeze(2).unsqueeze(3).unsqueeze(4), bias=None, stride=1, padding=0))
        #cam_refined = refine_cams(cam, (D, H, W), using_sigmoid=False)

        out1, out2 = self.act(out1), self.act(out2)
        if self.k is not None:
            out1, out2 = self.bilatreal_gate(out1, out2)

        return out1, out2, cam_l_refined, cam_r_refined










