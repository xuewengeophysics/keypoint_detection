# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 平行子网络信息多尺度融合模块
class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        """
        :param num_branches: 当前 stage 分支平行子网络的数目
        :param blocks: BasicBlock或者Bottleneck
        :param num_blocks: BasicBlock或者Bottleneck的数目

        :param num_inchannels: 输入通道数目
                    当stage = 2时： num_inchannels = [32, 64]
                    当stage = 3时： num_inchannels = [32, 64, 128]
                    当stage = 4时： num_inchannels = [32, 64, 128, 256]

        :param num_channels: 输出通道数目
                    当stage = 2时： num_inchannels = [32, 64]
                    当stage = 3时： num_inchannels = [32, 64, 128]
                    当stage = 4时： num_inchannels = [32, 64, 128, 256]

        :param fuse_method: 默认SUM
        :param multi_scale_output:
                    当stage = 2时： multi_scale_output=True
                    当stage = 3时： multi_scale_output=True
                    当stage = 4时： multi_scale_output=False
        """
        super(HighResolutionModule, self).__init__()

        # 对输入的一些参数进行检测
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        # 上面有详细介绍
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        # 为每个分支构建分支网络
        # 当stage=2,3,4时，num_branches分别为：2,3,4，表示每个stage平行网络的数目
        # 当stage=2,3,4时，num_blocks分别为：[4,4], [4,4,4], [4,4,4,4]，表示每个stage的每个平行分支BasicBlock或者Bottleneck的数目
        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        # 创建一个多尺度融合层，当stage=2,3,4时，len(self.fuse_layers)分别为2,3,4；
        # 其与num_branches在每个stage的数目是一致的
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    # # 判断num_branches（int）和num_blocks, num_inchannels, num_channels（list）
    # # 三者的长度是否一致，否则报错
    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    # 搭建1个分支，单个分支内部分辨率相等，1个分支由num_blocks[branch_index]个block组成，
    # block可以是两种ResNet模块中的一种；
    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        # 如果stride不为1，或者输入通道数目与输出通道数目不一致
        # 则通过卷积对其通道数进行改变
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        # 为当前分支branch_index创建一个block，此处进行下采样
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )

        # 把输出通道数，赋值给输入通道数，为下一stage作准备
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion

        # 为[1, num_blocks[branch_index]]分支创建block
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    # # 循环调用_make_one_branch函数创建多个分支
    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        # 循环为每个分支构建网络
        # 当stage=2,3,4时，num_branches分别为：2,3,4,表示每个stage平行网络的数目
        # 当stage=2,3,4时，num_blocks分别为：[4,4], [4,4,4], [4,4,4,4]，表示每个stage的每个平行分支BasicBlock或者Bottleneck的数目
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    # # 融合模块
    def _make_fuse_layers(self):
        # # (1)
        # # 如果分支数等于1，返回None，说明此时不需要使用融合模块；
        if self.num_branches == 1:
            return None

        # 平行子网络(分支)数目
        num_branches = self.num_branches
        # 输入通道数
        num_inchannels = self.num_inchannels
        fuse_layers = []
        # # (2)
        # # 双层循环
        # # 如果需要产生多分辨率的结果，就双层循环num_branches次；
        # # 如果只需要产生最高分辨率的结果，就将i确定为0；
        # 为每个分支都创建对应的特征融合网络，如果multi_scale_output==1，则只需要一个特征融合网络；
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                # # (2.1)
                # # 如果j>i，此时的目标是将所有分支上采样到和i分支相同的分辨率并融合；
                # # 也就是说j所代表的分支分辨率比i分支低；
                # # 2**(j-i)表示j分支上采样这么多倍才能和i分支分辨率相同。
                # # 先使用1x1卷积将j分支的通道数变得和i分支一致；
                # # 进而跟着BN；
                # # 然后依据上采样因子将j分支分辨率上采样到和i分支分辨率相同，此处使用最近邻插值；
                # 每个分支网络的输出有多钟情况
                # 1.当前分支信息传递到上一分支(即分辨率更高的分支，沿论文图示scale方向的反方向)的下一层(沿论文图示depth方向)，进行上采样，分辨率加倍
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )

                # # (2.2)
                # # 如果j=i，也就是说自身和自身之间不需要融合，nothing to do；
                # 2.当前分支信息传递到当前分支(即当前分支，沿论文图示scale方向的垂直方向)的下一层(沿论文图示depth方向)，不做任何操作，分辨率相同
                elif j == i:
                    fuse_layer.append(None)

                # # (2.3)
                # # 如果j<i，转换角色，此时最终目标是将所有分支采样到和i分支相同的分辨率并融合，
                # # 注意，此时j所代表的分支分辨率比i分支高，正好和(2.1)相反。
                # # 此时再次内嵌了一个循环，这层循环的作用是当(i-j)>1时，
                # # 也就是说两个分支的分辨率差了不止2倍，
                # # 此时还是2倍2倍往上采样，例如i-j=2时，j分支的分辨率比i分支大4倍，
                # # 就需要上采样2次，循环次数就是2；
                # 3.当前分支信息传递到下一分支(即分辨率更低的分支，沿论文图示scale方向)的下一层(沿论文图示depth方向)，分辨率减半
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        # # (2.3.1)
                        # # 当k==(i-j-1)时，举个例子，i=2，j=1，此时仅循环1次，并采用当前模块；
                        # # 此时直接将j分支使用3x3的步长为2的卷积下采样(不使用bias)，后接BN，不适用ReLU；
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        # # (2.3.2)
                        # # 当k!=(i-j-1)时，举个例子，i=3，j=1，此时循环2次，
                        # # 先采用当前模块；将j分支使用3x3的步长为2的卷积下采样(不使用bias)2倍，
                        # # 后接BN和ReLU，
                        # # 紧跟着再使用(2.3.1)中的模块，这是为了保证最后一次2倍下采样的卷积操作不使用ReLU，
                        # # 猜测也是为了保证融合后特征的多样性；
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    # # 前向传播函数，利用以上函数的功能搭建1个HighResolutionModule；
    def forward(self, x):
        # # (1)
        # # 当仅包含1个分支时，生成该分支，没有融合模块，直接返回；
        # 当stage=2,3,4时，num_branches分别为：2,3,4，表示每个stage平行网络的数目
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        # # (2)
        # # 当包含不仅1个分支时，先将对应分支的输入特征输入到对应分支，得到对应分支的输出特征；
        # # 紧接着执行融合模块；
        # # (2.1)
        # # 循环将对应分支的输入特征输入到对应分支模型中，得到对应分支的输出特征；
        # 当前有多少个网络分支，则有多少个x当作输入
        # 当stage=2：x=[b,32,64,48],[b,64,32,24]
        #           -->[b,32,64,48],[b,64,32,24]
        # 当stage=3：x=[b,32,64,48],[b,64,32,24],[b,128,16,12]
        #           -->[b,32,64,48],[b,64,32,24],[b,128,16,12]
        # 当stage=4：x=[b,32,64,48],[b,64,32,24],[b,128,16,12],[b,256,8,6]
        #           -->[b,32,64,48],[b,64,32,24],[b,128,16,12],[b,256,8,6]
        # 简单的说，该处就是对每个分支进行了BasicBlock或者Bottleneck操作
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        # # (2.2)
        # # 融合模块
        # # 每次多尺度之间的加法运算都是从最上面的尺度开始往下加，
        # # 所以y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])；
        # # 加到它自己的时候，不需要经过融合函数的处理，直接加；
        # # 遇到不是最上面的尺度那个特征图或者它本身相同分辨率的那个特征图时，
        # # 需要经过融合函数处理再加；
        # # 最后将ReLU激活后的融合(加法)特征append到x_fuse，
        # # x_fuse的长度等于1(单尺度输出)或者num_branches(多尺度输出)

        x_fuse = []
        # 对每个分支进行融合(信息交流)
        for i in range(len(self.fuse_layers)):
            # 循环融合多个分支的输出信息，当作输入，进行下一轮融合
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA'] # _C.MODEL.EXTRA = CN(new_allowed=True)，whether adding new key is allowed when merging with other configs.
        super(PoseHighResolutionNet, self).__init__()


        # stem net
        # 进行一系列的卷积操作，获得最初始的特征图N11
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)


        # 获取stage2的相关配置信息
        self.stage2_cfg = extra['STAGE2']
        # num_channels=[32, 64]，表示输出通道；
        # 32是高分辨率平行分支N21的输出通道数；
        # 64是新建平行分支N22的输出通道数；
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        # 这里的block为BASIC
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        # block.expansion默认为1，num_channels表示输出通道[32, 64]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        # 这里会生成新的平行分支N2网络，即N11-->N21,N22这个过程
        # 同时会对输入的特征图x进行通道变换(如果输入输出通道数不一致)
        self.transition1 = self._make_transition_layer([256], num_channels)
        # 对平行子网络进行加工，让其输出的y，可以当作下一个stage的输入x，
        # 这里的pre_stage_channels为当前stage的输出通道数，也就是下一个stage的输入通道数
        # 同时平行子网络信息交换模块也包含再其中
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)


        # 获取stage3的相关配置信息
        self.stage3_cfg = extra['STAGE3']
        # num_channels=[32, 64, 128]，表示输出通道；
        # 32是高分辨率平行分支N31的输出通道数；
        # 64是平行分支N32的输出通道数；
        # 128是新建平行分支N33的输出通道数
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        # 这里的block为BasicBlock
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        # block.expansion默认为1，num_channels表示输出通道[32, 64, 128]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        # 这里会生成新的平行分支N3网络，即N22-->N32,N33这个过程
        # 同时会对输入的特征图x进行通道变换(如果输入输出通道数不一致)
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        # 对平行子网络进行加工，让其输出的y，可以当作下一个stage的输入x，
        # 这里的pre_stage_channels为当前stage的输出通道数，也就是下一个stage的输入通道数
        # 同时平行子网络信息交换模块，也包含再其中
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)


        # 获取stage4的相关配置信息
        self.stage4_cfg = extra['STAGE4']
        # num_channels=[32, 64, 128, 256]，表示输出通道；
        # 32是高分辨率平行分支N41的输出通道数；
        # 64是平行分支N42的输出通道数；
        # 128是平行分支N43的输出通道数；
        # 256是新建平行分支N44的输出通道数；
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        # 这里的block为BasicBlock；
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        # block.expansion默认为1，num_channels表示输出通道[32, 64, 128, 256]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        # 这里会生成新的平行分支N4网络，即N33-->N43,N44这个过程
        # 同时会对输入的特征图x进行通道变换(如果输入输出通道数不一致)
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        # 对平行子网络进行加工，让其输出的y，可以当作下一个stage的输入x，
        # 这里的pre_stage_channels为当前stage的输出通道数，也就是下一个stage的输入通道数
        # 同时平行子网络信息交换模块，也包含再其中
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)


        # 对最终的特征图混合之后进行一次卷积, 预测人体关键点的heatmap
        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg['MODEL']['NUM_JOINTS'], # 17个关键点，通道数为17层，每一层对应1个关键点的heatmap
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

    # 创建新的平行子分支网络
    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        """
        :param num_channels_pre_layer: 上一个stage平行网络的输出通道数目，为一个list,
            stage=2时, num_channels_pre_layer=[256]
            stage=3时, num_channels_pre_layer=[32,64]
            stage=4时, num_channels_pre_layer=[32,64,128]
        :param num_channels_cur_layer: 当前stage平行网络的输出通道数目，为一个list,
            stage=2时, num_channels_cur_layer = [32,64]
            stage=3时, num_channels_cur_layer = [32,64,128]
            stage=4时, num_channels_cur_layer = [32,64,128,256]
        """
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        # 对stage的每个分支进行处理
        for i in range(num_branches_cur):
            # 如果不是最后一个分支
            if i < num_branches_pre:
                # 如果当前层的输入通道和输出通道数不相等，则通过卷积对通道数进行变换
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                # 如果当前层的输入通道和输出通道数相等，则什么都不做
                else:
                    transition_layers.append(None)

            # 如果是最后一个分支，则再新建一个分支（该分支分辨率会减少一半）
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    # block: BasicBlock或Bottleneck
    # blocks: 块体个数
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # 构建论文中平行子网络信息交流的模块
    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        """
        当stage=2时： num_inchannels=[32,64]           multi_scale_output=Ture
        当stage=3时： num_inchannels=[32,64,128]       multi_scale_output=Ture
        当stage=4时： num_inchannels=[32,64,128,256]   multi_scale_output=False
        """
        # 当stage=2,3,4时，num_modules分别为：1,4,3
        # 表示HighResolutionModule（平行之网络交换信息模块）模块的数目
        num_modules = layer_config['NUM_MODULES']
        # 当stage=2,3,4时，num_branches分别为：2,3,4
        # 表示每个stage平行网络的数目
        num_branches = layer_config['NUM_BRANCHES']
        # 当stage=2,3,4时，num_blocks分别为：[4,4], [4,4,4], [4,4,4,4]
        # 表示每个stage blocks(BasicBlock或者Bottleneck)的数目
        num_blocks = layer_config['NUM_BLOCKS']
        # 当stage=2,3,4时，num_channels分别为：[32,64], [32,64,128], [32,64,128,256]
        num_channels = layer_config['NUM_CHANNELS']
        # 当stage=2,3,4时，block分别为：BasicBlock,BasicBlock,BasicBlock
        block = blocks_dict[layer_config['BLOCK']]
        # 当stage=2,3,4时，都为SUM，表示特征融合的方式
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        # 根据num_modules的数目创建HighResolutionModule
        for i in range(num_modules):
            # multi_scale_output is only used last module
            # multi_scale_output 只被用在最后一个HighResolutionModule
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            # 根据参数，添加HighResolutionModule
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            # 获得最后一个HighResolutionModule的输出通道数
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        # 对应论文中的stage1
        # # 经过一系列的卷积，获得初步特征图，总体过程为x[b, 3, 256, 192]-->x[b, 256, 64, 48]
        x = self.conv1(x)  # x[b,   3, 256, 192] --> x[b,  64, 128,  96]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)  # x[b,  64, 128,  96] --> x[b,  64,  64,  48]
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x) # x[b,  64,  64,  48] --> x[b, 256,  64,  48]


        # 对应论文中的stage2
        # 其中包含了创建分支的过程，即 N11-->N21,N22 这个过程
        # N22的分辨率为N21的二分之一，总体过程为:
        # x[b,256,64,48] --> y[b, 32, 64, 48]  因为通道数不一致，通过卷积进行通道数变换
        #                    y[b, 64, 32, 24]  通过新建平行分支生成
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        # 总体过程如下(经过一些卷积操作，但是特征图的分辨率和通道数都没有改变)：
        # x[b, 32, 64, 48] -->  y[b, 32, 64, 48]
        # x[b, 64, 32, 24] -->  y[b, 64, 32, 24]
        y_list = self.stage2(x_list)


        # 对应论文中的stage3
        # 其中包含了创建分支的过程，即 N22-->N32,N33 这个过程
        # N32的分辨率为N31的二分之一，
        # N33的分辨率为N32的二分之一，
        # y[b, 32, 64, 48] --> x[b, 32,  64, 48]   因为通道数一致，没有做任何操作
        # y[b, 64, 32, 24] --> x[b, 64,  32, 24]   因为通道数一致，没有做任何操作
        #                      x[b, 128, 16, 12]   通过新建平行分支生成
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        # 总体过程如下(经过一些卷积操作,但是特征图的分辨率和通道数都没有改变)：
        # x[b, 32, 64, 48] --> x[b, 32, 64, 48]
        # x[b, 32, 32, 24] --> x[b, 32, 32, 24]
        # x[b, 64, 16, 12] --> x[b, 64, 16, 12]
        y_list = self.stage3(x_list)


        # 对应论文中的stage4
        # 其中包含了创建分支的过程，即 N33-->N43,N44 这个过程
        # N42的分辨率为N41的二分之一
        # N43的分辨率为N42的二分之一
        # N44的分辨率为N43的二分之一
        # y[b,  32,  64,  48] --> x[b,  32,  64,  48]  因为通道数一致，没有做任何操作
        # y[b,  64,  32,  24] --> x[b,  64,  32,  24]  因为通道数一致，没有做任何操作
        # y[b, 128,  16,  12] --> x[b, 128,  16,  12]  因为通道数一致，没有做任何操作
        #                         x[b, 256,   8,   6]  通过新建平行分支生成
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        # 进行多尺度特征融合
        # x[b,  32,  64,  48] -->
        # x[b,  64,  32,  24] -->
        # x[b, 128,  16,  12] -->
        # x[b, 256,   8,   6] --> y[b,  32,  64,  48]
        y_list = self.stage4(x_list)

        # y[b,  32,  64,  48] --> x[b,  17,  64,  48]
        x = self.final_layer(y_list[0])

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
