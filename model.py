from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from util import IMG_HEIGHT, IMG_WIDTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2 基本卷积组件'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    """
    上采样层 先上采样，再填充，然后通道维跟其他特征图连接，最后经过一个标准卷积层。
    """

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)  # 对x1进行双线性插值上采样

        # input is BCHW，根据特征图x1和x2的尺寸差异，对x1进行填充。
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        if x2 is not None:  # 将x2、x1的通道维进行连接
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)  # 连接之后通过一个标准的卷积层
        return x


def get_mesh(batch_size, shape_x, shape_y):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mesh = torch.cat([torch.tensor(mg_x).to(device), torch.tensor(mg_y).to(device)], 1)
    return mesh


class Model(nn.Module):
    '''Mixture of previous classes'''

    def __init__(self, n_classes):
        super(MyUNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')

        self.conv0 = double_conv(5, 64)
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 512)
        self.conv3 = double_conv(512, 1024)

        self.mp = nn.MaxPool2d(2)

        self.up1 = up(1282 + 1024, 512)
        self.up2 = up(512 + 512, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        x0 = torch.cat([x, mesh1], 1)
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))

        x_center = x[:, :, :, IMG_WIDTH // 8: -IMG_WIDTH // 8]
        feats = self.base_model.extract_features(x_center)
        bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(device)
        feats = torch.cat([bg, feats, bg], 3)

        mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3])
        feats = torch.cat([feats, mesh2], 1)

        # 先将特征采样，然后和某些特征相连接，然后然后再上采样，再和某些特征相连，最后通过一个卷积层得到预测输出。
        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.outc(x)
        # x[:, 0] = self.sigmoid(x[:, 0])
        # x_classification = self.outc_classification(x)
        # x_regression = self.outc_regression(x)
        # x = torch.cat([x_classification, x_regression], 1)
        return x


# 损失函数的设定。work的是不normal的分类损失加上normal的回归损失。
def criterion(prediction, mask, gaussian_mask, regr, size_average=True, split_loss=False):
    pred_mask = torch.sigmoid(prediction[:, 0])  # 将预测的值通过sigmoid函数，得到二分类置信度。

    # Binary mask loss
    # mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    # mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    # # mask_loss = -mask_loss.mean(0).sum()

    # focal loss
    # alpha = 0.90
    # beta = 2
    # mask_focal_loss = mask * alpha * ((1 - pred_mask) ** beta) * torch.log(pred_mask + 1e-12) + (1 - mask) * (
    #         1 - alpha) * (pred_mask ** beta) * torch.log(1 - pred_mask + 1e-12)
    # mask_focal_loss = -mask_focal_loss.mean(0).sum()  # 该损失是在整个图上计算的，然后在batch维上取平均，最后相加，而不是每个点的分类损失。

    # gaussian focal loss (heatmap)
    alpha = 2  # 用来减少易分类的点的损失，加大难分类点的损失。
    beta = 4  # 用来处理高斯标签。
    gamma = 0.75  # 用来处理正负样本的不平衡,加大正样本的比重。
    gaussian_focal_loss = -((((1 - pred_mask[gaussian_mask == 1]) ** alpha) * torch.log(
        pred_mask[gaussian_mask == 1] + 1e-12)).sum() * gamma + \
                            (((1 - gaussian_mask) ** beta) * (pred_mask ** alpha) * torch.log(
                                1 - pred_mask + 1e-12)).sum() * (1 - gamma))

    # gaussian_focal_loss /= mask.sum()  # 此时不用对每个点loss进行normal
    gaussian_focal_loss /= prediction.shape[0]  # 在整个batch维上取平均

    # Regression L1 loss 回归损失。
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)  # 回归损失
    regr_loss = regr_loss.mean(0)  # 在batch上将回归损失平均。

    lam = 2  #
    regr_loss = lam * regr_loss
    loss = gaussian_focal_loss + regr_loss
    if not size_average:
        loss *= prediction.shape[0]

    if split_loss:
        return loss, gaussian_focal_loss, regr_loss
    else:
        return loss
