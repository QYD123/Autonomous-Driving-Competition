# coding=UTF-8
import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from torch.utils.data import DataLoader
import torch
from model import Model
from util import extract_coords, coords2str  # 导入函数
from util import PATH, train, test  # 导入常量（路径、训练集、测试集DataFrame对象）
from dataset import CarDataset
from model import criterion
from mAP import calculate_mAP

import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 建立一个DataFrame对象，每一行包含每辆车的坐标和姿态信息。


train_images_dir = PATH + '/train_images/{}.jpg'
test_images_dir = PATH + '/test_images/{}.jpg'
df_train, df_dev = train_test_split(train, test_size=0.1,
                                    random_state=42)  # 划分训练集和验证集
df_test = test
df_dev_1 = df_dev.copy()
df_dev_pred_1 = pd.DataFrame()
df_dev_pred_2 = pd.DataFrame()
df_dev_pred_1['ImageId'] = df_dev_1['ImageId']
df_dev_pred_1['PredictionString'] = [''] * len(df_dev_1)
df_dev_pred_2['ImageId'] = df_dev_1['ImageId']
df_dev_pred_2['PredictionString'] = [''] * len(df_dev_1)
# Create dataset objects 创建数据集对象
train_dataset = CarDataset(df_train, train_images_dir, training=True)
dev_dataset = CarDataset(df_dev, train_images_dir, training=False)
test_dataset = CarDataset(df_test, test_images_dir, training=False)

# 数据集的设置
BATCH_SIZE = 4
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
n_epochs = 30

# 模型和学习率的设置
model = Model(8).to(device)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(n_epochs, 10) * len(train_loader) // 3, gamma=0.1)


def train_model(epoch, history=None):
    model.train()
    loss_all = 0
    mask_focal_loss_all = 0
    regr_loss_all = 0
    for batch_idx, (img_batch, mask_batch, gaussian_mask_batch, regr_batch) in enumerate(
            tqdm(train_loader, desc="训练中")):
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        gaussian_mask_batch = gaussian_mask_batch.to(device)
        regr_batch = regr_batch.to(device)

        optimizer.zero_grad()
        output = model(img_batch)

        loss, mask_focal_loss, regr_loss = criterion(output, mask_batch, gaussian_mask_batch, regr_batch,
                                                     split_loss=True)
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()

        loss.backward()
        loss_all += loss.data
        mask_focal_loss_all += mask_focal_loss.data
        regr_loss_all += regr_loss.data

        optimizer.step()
        exp_lr_scheduler.step()  # 学习率衰减

    print('\nTrain Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}\tbinary loss {:.6f}\tregression loss {:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss_all / len(train_loader), mask_focal_loss_all / len(train_loader),
        regr_loss_all / len(train_loader)))


def evaluate_model(epoch, history=None):
    model.eval()
    loss = 0
    dev_pred_1 = []  # logtic 阈值0.3
    dev_pred_2 = []  # logtic 阈值0.6
    with torch.no_grad():  # try finally的简写形式
        for img_batch, mask_batch, gaussian_batch, regr_batch in tqdm(dev_loader, desc="验证中"):
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            gaussian_batch = gaussian_batch.to(device)
            regr_batch = regr_batch.to(device)

            output = model(img_batch)

            loss += criterion(output, mask_batch, gaussian_batch, regr_batch, size_average=False).data
            output = output.data.cpu().numpy()
            for out in output:
                coords_1 = extract_coords(out, threshold=0.5)
                coords_2 = extract_coords(out, threshold=0.6)
                s_1 = coords2str(coords_1)
                s_2 = coords2str(coords_2)
                dev_pred_1.append(s_1)
                dev_pred_2.append(s_2)

    loss /= len(dev_loader.dataset)
    df_dev_pred_1['PredictionString'] = dev_pred_1
    df_dev_pred_2['PredictionString'] = dev_pred_2

    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()

    print('Dev loss: {:.4f}'.format(loss))
    mAP_1 = calculate_mAP(valid_df=df_dev_pred_1, train_df=df_dev)
    mAP_2 = calculate_mAP(valid_df=df_dev_pred_2, train_df=df_dev)
    print('mAP threshold: 0.5:', mAP_1)
    print('mAP threshold: 0.6:', mAP_2)
    mAP_history.loc[epoch, '0.5'] = mAP_1
    mAP_history.loc[epoch, '0.6'] = mAP_2


if __name__ == '__main__':

    # model.load_state_dict(torch.load("17_model.pth"))
    # history = pd.DataFrame()
    # mAP_list = []
    # mAP_history = pd.DataFrame()
    # # 训练和验证
    # for epoch in range(18, n_epochs):
    #     tic = time.time()
    #     print("第{}轮训练开始".format(epoch))
    #     torch.cuda.empty_cache()  # 清空GPU缓存
    #     train_model(epoch, history)
    #     if epoch > 10:
    #         evaluate_model(epoch, history)
    #     toc = time.time()
    #     print("第{}轮训练和验证结束, 耗时{}min".format(epoch, (toc - tic) // 60))
    #     if (epoch + 1) % 1 == 0:
    #         torch.save(model.state_dict(), './{}_model.pth'.format(epoch))
    #     gc.collect()  # 内存垃圾回收
    # history.to_csv('loss_history.csv')
    # mAP_history.to_csv('mAP_history.csv')

    # # 找到mAP最大的值所对应的轮数
    # mAP_list_max = max(mAP_list)
    # index = mAP_list.index(mAP_list_max)
    # model.load_state_dict(torch.load('{}_model.pth'.format(index)))
    model.load_state_dict(torch.load("21_model.pth"))
    predictions = []
    model.eval()
    #
    # # 模型的推理，得到输出的数据。
    for img, _, _, _ in tqdm(test_loader, desc='预测中'):  # tqdm只需关注其能加进度条就行了，其它的不用管。
        with torch.no_grad():
            output = model(img.to(device))
        output = output.data.cpu().numpy()
        for out in output:  # 每个batch中提取单独的一个预测输出
            coords = extract_coords(out, threshold=0.5)
            s = coords2str(coords)
            predictions.append(s)

    test['PredictionString'] = predictions
    test.to_csv('predictions.csv', index=False)
    print(test.head())
