from torch.utils.data import Dataset
import torch
import numpy as np
from util import imread
import cv2
from util import IMG_WIDTH, IMG_HEIGHT
from util import get_mask_and_regr


# 数据增强
def preprocess_image(img, flip=False):
    img = img[img.shape[0] // 2:]  # 裁掉上半部分的图像，（通常没有车辆）
    bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, :img.shape[1] // 6]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if flip:
        img = img[:, ::-1]  # 直接对图像水平翻转
    return (img / 255).astype('float32')  # 把图像的格式转换到0到1之间


class CarDataset(Dataset):
    """Car dataset."""

    def __init__(self, dataframe, root_dir, training=True, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image name
        idx, labels = self.df.values[idx]  # 得到了CSV文件中按照顺序排列的图片的ID和其标签
        img_name = self.root_dir.format(idx)  # 由图像的ID得到其路径

        # Augmentation
        flip = False
        shift_rgb = False
        if self.training:
            flip = np.random.randint(5) == 1  # 设定一定的翻转概率
            shift_rgb = np.random.randint(5) == 1  # 设定一定的RGB shift 概率
        # Read image
        img0 = imread(img_name, True)
        img = preprocess_image(img0, flip=flip)
        if shift_rgb:
            img = img[:, :, ::-1].copy()
        img = np.rollaxis(img, 2, 0)

        # Get mask and regression maps
        mask, gaussian_mask, regr = get_mask_and_regr(img0, labels, flip=flip)
        regr = np.rollaxis(regr, 2, 0)  # convert HWC to CHW

        return [img, mask, gaussian_mask, regr]
