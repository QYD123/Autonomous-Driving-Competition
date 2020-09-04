from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from multiprocessing import Pool
from math import acos, pi


def expand_df(df, PredictionStringCols):
    df = df.dropna().copy()
    # 清洗数据
    df['NumCars'] = [int((x.count(' ') + 1) / 7) for x in df['PredictionString']]
    # 每张图片中汽车数量=间隔出现的次数+1再除以7

    image_id_expanded = [item for item, count in zip(df['ImageId'], df['NumCars']) for i in range(count)]  # 嵌套列表生成式
    # 图片id扩充=每个图片的车辆名字都叫图片的
    prediction_strings_expanded = df['PredictionString'].str.split(' ', expand=True).values.reshape(-1, 7).astype(float)
    # 每张图中把每一辆车的都切开，变成矩阵
    prediction_strings_expanded = prediction_strings_expanded[~np.isnan(prediction_strings_expanded).all(axis=1)]
    # 去掉缺失值

    df = pd.DataFrame(
        {
            'ImageId': image_id_expanded,
            PredictionStringCols[0]: prediction_strings_expanded[:, 0],
            PredictionStringCols[1]: prediction_strings_expanded[:, 1],
            PredictionStringCols[2]: prediction_strings_expanded[:, 2],
            PredictionStringCols[3]: prediction_strings_expanded[:, 3],
            PredictionStringCols[4]: prediction_strings_expanded[:, 4],
            PredictionStringCols[5]: prediction_strings_expanded[:, 5],
            PredictionStringCols[6]: prediction_strings_expanded[:, 6]
        })
    return df
    # 重新定义df表格，分类，加表头


# 预测一大串字符串变成一个字典
def str2coords(s, names):
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
    return coords


# 计算距离差距，abs_dist指绝对误差还是相对真实值的相对误差
def TranslationDistance(p, g, abs_dist=False):
    dx = p['x'] - g['x']
    dy = p['y'] - g['y']
    dz = p['z'] - g['z']
    diff0 = (g['x'] ** 2 + g['y'] ** 2 + g['z'] ** 2) ** 0.5
    diff1 = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
    if abs_dist:
        diff = diff1
    else:
        diff = diff1 / diff0
    return diff


# 计算角度差距
def RotationDistance(p, g):
    true = [g['pitch'], g['yaw'], g['roll']]
    pred = [p['pitch'], p['yaw'], p['roll']]
    q1 = R.from_euler('xyz', true)  # 输入轴角
    q2 = R.from_euler('xyz', pred)
    diff = R.inv(q2) * q1
    W = np.clip(diff.as_quat()[-1], -1., 1.)  # 四元数

    # in the official metrics code:
    # https://www.kaggle.com/c/pku-autonomous-driving/overview/evaluation
    #   return Object3D.RadianToDegree( Math.Acos(diff.W) )
    # this code treat θ and θ+2π differntly.
    # So this should be fixed as follows.
    W = (acos(W) * 360) / pi
    if W > 180:
        W = 360 - W
    return W


# 平移和角度的阈值
thres_tr_list = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
thres_ro_list = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]


def check_match(idx):
    idx, train_df, valid_df = idx
    train_df = train_df.dropna().copy()
    valid_df = valid_df.dropna().copy()
    # idx 是取相应的
    keep_gt = False
    thre_tr_dist = thres_tr_list[idx]
    thre_ro_dist = thres_ro_list[idx]

    # 字典生成式（外层：imgID，[内层：一个name，对应一个数据,循环对应，直到一个图片结束]）训练的字典和验证的字典
    train_dict = {imgID: str2coords(s, names=['carid_or_score', 'pitch', 'yaw', 'roll', 'x', 'y', 'z']) for imgID, s
                  in
                  zip(train_df['ImageId'], train_df['PredictionString'])}
    valid_dict = {imgID: str2coords(s, names=['pitch', 'yaw', 'roll', 'x', 'y', 'z', 'carid_or_score']) for imgID, s
                  in
                  zip(valid_df['ImageId'], valid_df['PredictionString'])}
    result_flg = []  # 1 for TP, 0 for FP
    scores = []
    MAX_VAL = 10 ** 10
    for img_id in valid_dict:  # 从预测中拿到key值（图片ID值）
        # 一个图片一个图片找
        for pcar in sorted(valid_dict[img_id], key=lambda x: -x['carid_or_score']):
            # 一张图片中的每辆车的属性存储到一个字典当中，然后将每个字典按照置信度排序。
            # find nearest GT
            min_tr_dist = MAX_VAL  # 初始阈值
            min_idx = -1  # 没有找到id记为-1
            for idx, gcar in enumerate(train_dict[img_id]):
                # 带编号迭代（enumerate自动为图片中的车加编号）
                tr_dist = TranslationDistance(pcar, gcar)  # 计算距离
                # 满足距离之后，在进行角度筛选
                if tr_dist < min_tr_dist:
                    min_tr_dist = tr_dist  # 找到离该辆车距离最小的距离
                    min_ro_dist = RotationDistance(pcar, gcar)  # 计算角度
                    min_idx = idx  # 匹配成功之后，车在标签字典中的序号。

            # set the result
            if min_tr_dist < thre_tr_dist and min_ro_dist < thre_ro_dist:
                # 如果都满足阈值
                if not keep_gt:
                    train_dict[img_id].pop(min_idx)  # 从标签字典中删除该车。
                result_flg.append(1)  # 找到了,且满足阈值：1（TP）
            else:
                result_flg.append(0)  # 找到了,但是不满足阈值：0 (FP)
            scores.append(pcar['carid_or_score'])

    return result_flg, scores
    # 返回:result_flg是这个进程所有图片，所有车辆的TP&FP（1，0或者无）,scores是所有置信度


def calculate_AP(precision_list, recall_list):
    AP = 0
    for i in range(len(precision_list) - 1):
        AP += ((recall_list[i + 1] - recall_list[i]) * precision_list[i + 1])
    return AP


def calculate_mAP(valid_df, train_df):
    # 主函数：
    # validation_prediction = df_dev
    # valid_df = pd.read_csv('val_predictions.csv')  # 输入接口 1
    # expanded_valid_df = expand_df(valid_df, ['pitch', 'yaw', 'roll', 'x', 'y', 'z', 'Score'])  # 预测的
    valid_df = valid_df.fillna('')

    # train_df = pd.read_csv('../input/pku-autonomous-driving/train.csv')  # 输入接口 2
    train_df = train_df[train_df.ImageId.isin(valid_df.ImageId.unique())]  # 取出正确的
    # data description page says, The pose information is formatted as
    # model type, yaw, pitch, roll, x, y, z
    # but it doesn't, and it should be
    # model type, pitch, yaw, roll, x, y, z
    expanded_train_df = expand_df(train_df, ['model_type', 'pitch', 'yaw', 'roll', 'x', 'y', 'z'])  # 给正确的也加标题

    max_workers = 10  # 最大进程
    n_gt = len(expanded_train_df)  # 标签中的总车辆数(TP+FP+FN)
    ap_list = []
    p = Pool(processes=max_workers)  # 创建多进程对象
    # confidence_list = [i / 10 for i in range(10, 2, -1)]  # 先去掉重复的阈值，然后将阈值从小到大排列。

    precision_list = []
    recall_list = []
    for result_flg, scores in p.imap(check_match,
                                     zip(range(10), [train_df] * 10, [valid_df] * 10)):  # 采用多进程加快运行的速度,返回一个list
        if np.sum(result_flg) > 0:
            n_tp = np.sum(result_flg)
            recall = n_tp / n_gt  # recall: TP/(TP+FP+FN)
            precision = n_tp / len(result_flg)  # precision: TP/(TP+FP)
            ap = precision * recall  # 这样算的ap只有一个点，并不是常规的ap值。
            # ap = average_precision_score(result_flg, scores) * recall
            # average_precision_score详见：http://yongyuan.name/blog/evaluation-of-information-retrieval.html
            # Precesion=recall
            # ap= Precesion
        else:
            ap = 0
        ap_list.append(ap)
    map = np.mean(ap_list)  # 所有阈值取平均
    return map

# mAP理论:https://blog.csdn.net/Katherine_hsr/article/details/79266880
# 目标检测、图像分类中的模型效果指标precision与recall的本质探索：https://blog.csdn.net/weixin_37340613/article/details/88841261
# mAP（mean average precision）平均精度均值：https://www.jianshu.com/p/82be426f776e
