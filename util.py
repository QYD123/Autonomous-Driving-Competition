import numpy as np
import cv2
from math import sin, cos
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize
import pandas as pd
from math import sqrt
import torch


# 读取图像
def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])  # 将opencv读出的BGR图转换成RGB图的格式。
    return img


# 常量的设置
PATH = "/root/Self-Driving/Data/"  # 加r,告诉编译器这个字符串是个raw string,忽略斜杠的转义作用。
img = imread(PATH + "train_images/ID_8a6e65317" + '.jpg')
IMG_SHAPE = img.shape
# 图像的设置
IMG_WIDTH = 2500
IMG_HEIGHT = IMG_WIDTH // 16 * 5
# 图像尺寸的设置 centerpoint - resnet
# IMG_WIDTH = 2048
# IMG_HEIGHT = 512
MODEL_SCALE = 8
train = pd.read_csv(PATH + 'train.csv')  # 读取csv文件，返回一个DataFrame对象。
test = pd.read_csv(PATH + 'sample_submission.csv')
# 定义相机的内参矩阵，该矩阵表示从相机坐标系到像素坐标系下的转换。
camera_intrinsic_matrix = np.array([[2304.5479, 0, 1686.2379],
                                    [0, 2305.8757, 1354.9849],
                                    [0, 0, 1]], dtype=np.float32)
camera_intrinsic_matrix_inv = np.linalg.inv(camera_intrinsic_matrix)

DISTANCE_THRESH_CLEAR = 2  # 非极大值抑制的阈值


# 从训练数据中的字符串类型中得到车辆的三个坐标值和单个姿态值
def str2coords(s, names=('model_type', 'yaw', 'pitch', 'roll', 'x', 'y', 'z')):
    """
     Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    """

    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['model_type'] = int(coords[-1]['model_type'])
    return coords


points_df = pd.DataFrame()
for col in ['x', 'y', 'z', 'yaw', 'pitch', 'roll']:
    arr = []
    for ps in train['PredictionString']:
        coords = str2coords(ps)
        arr += [c[col] for c in coords]
    points_df[col] = arr

xzy_slope = LinearRegression()
X = points_df[['x', 'z']]
y = points_df['y']
xzy_slope.fit(X, y)


# 从字符串中读入数据，得到车辆在像素坐标系下的位置坐标。
def get_img_coords(s):
    """
        从字符串中读入数据，得到车辆在像素坐标系点的坐标。
        Input is a PredictionString (e.g. from train dataframe)
    Output is two arrays:
        xs: x coordinates in the image (row)
        ys: y coordinates in the image (column)
    """

    coords = str2coords(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T  # 得到一个 array，
    img_p = np.dot(camera_intrinsic_matrix, P).T  # 转置得到相应的输出
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2]  # z = Distance from the camera
    return img_xs, img_ys


# convert euler angle to rotation matrix 由欧拉角计算旋转矩阵。
def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))


# 传参angle = pi的话，首次调用将2*n*pi+k(k为锐角)这样的角度，变成k-pi，在区间[-pi,pi]之间的形式；再次调用，返回k。
def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x


# 将标签的值处理成回归预测（网络输出）的量的格式。
def _regr_preprocess(regr_dict, flip=False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            regr_dict[k] = -regr_dict[k]
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] / 100  # 对于标签数据的预处理，此处对x、y、z除了一个100，所以在后期处理的时候给乘了回来，归一化其度量。
    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])
    regr_dict.pop('pitch')
    regr_dict.pop('model_type')
    return regr_dict


# 从回归预测的得到需要初始标签的形式。
def _regr_back(regr_dict):
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] * 100  # x、y、z乘100，变回去。
    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)  # 滚转角变回K
    # 车辆的俯仰角由sin、cos变回其角度。
    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin'] ** 2 + regr_dict['pitch_cos'] ** 2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin'] ** 2 + regr_dict['pitch_cos'] ** 2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict


def calculate_gaussian(gaussian_mask, center_x, center_y):
    height, width = gaussian_mask.shape
    sigma = 1 / 6 * sqrt(3 ** 2 + 3 ** 2)
    for x in range(center_x - 1 if (center_x - 1) > 0 else 0, center_x + 2 if (center_x + 2 < height) else height):
        for y in range(center_y - 1 if (center_y - 1) > 0 else 0,
                       center_y + 2 if (center_y + 2 < width) else width):
            tmp = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * (sigma ** 2)))
            if tmp > gaussian_mask[x, y]:
                gaussian_mask[x, y] = tmp
    return gaussian_mask


# 由标签得到mask和regr(回归预测的置信度标签和其它量的标签) 处理标签
def get_mask_and_regr(img, labels, flip=False):
    mask = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE], dtype='float32')
    gaussian_mask = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE], dtype='float32')
    regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    regr = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 7],
                    dtype='float32')  # 一个特征图，第三维的长度是7，分别是7个预测的量，这是回归预测量的标签。
    coords = str2coords(labels)  # 返回一个列表，列表中包含车辆数目的字典，字典中的键值是参数['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    xs, ys = get_img_coords(labels)  # 从图片对应的标签中，得到不同的车辆在图像坐标系下的坐标。所谓高斯标签应该只是来调整这种对应关系的错误才对。
    for x, y, regr_dict in zip(xs, ys,
                               coords):  # 得到标签在图像坐标系下的坐标和对应的属性字典，字典中的键值是参数['x', 'y', 'z', 'yaw', 'pitch', 'roll']
        x, y = y, x  # 对x、y的值进行交换，将坐标转换到像素坐标系下。
        x = (x - img.shape[0] // 2) * IMG_HEIGHT / (img.shape[0] // 2) / MODEL_SCALE  # 对像素坐标处理，将原始的坐标处理成网络输出的特征图中的标签。
        x = np.round(x).astype('int')
        y = (y + img.shape[1] // 6) * IMG_WIDTH / (img.shape[1] * 4 / 3) / MODEL_SCALE  # 可以改变的对图像的预处理以及在网络中前向传播损失的尺寸。
        y = np.round(y).astype('int')
        if x >= 0 and x < IMG_HEIGHT // MODEL_SCALE and y >= 0 and y < IMG_WIDTH // MODEL_SCALE:  # 表示其处理的范围
            mask[x, y] = 1  # 普通标签
            gaussian_mask = calculate_gaussian(gaussian_mask, x, y)  # 高斯标签，标签所在的概率为1，其余的位置的概率呈现高斯核的形状。
            regr_dict = _regr_preprocess(regr_dict, flip)  # 对标签字典中的值进行处理，换算成最终输出的值。
            regr[x, y] = [regr_dict[n] for n in sorted(regr_dict)]  # 将回归预测得到的字典的列表对应到像素坐标上去

    if flip:
        mask = np.array(mask[:, ::-1])
        gaussian_mask = np.array(gaussian_mask[:, ::-1])
        regr = np.array(regr[:, ::-1])
    return mask, gaussian_mask, regr


def convert_3d_to_2d(x, y, z, fx=2304.5479, fy=2305.8757, cx=1686.2379, cy=1354.9849):
    return x * fx / z + cx, y * fy / z + cy


# 优化求解x、y、z的值。
def optimize_xy(r, c, x0, y0, z0, flipped=False):
    def distance_fn(xyz):
        x, y, z = xyz
        xx = -x if flipped else x
        slope_err = (xzy_slope.predict([[xx, z]])[0] - y) ** 2  # 通过前面拟合的通过x、y坐标来预测z坐标的线性回归函数来计算误差。
        x, y = convert_3d_to_2d(x, y, z)  # 从3d坐标到像素坐标系下的坐标。
        y, x = x, y
        x = (x - IMG_SHAPE[0] // 2) * IMG_HEIGHT / (IMG_SHAPE[0] // 2) / MODEL_SCALE
        y = (y + IMG_SHAPE[1] // 6) * IMG_WIDTH / (IMG_SHAPE[1] * 4 / 3) / MODEL_SCALE
        return max(0.2, (x - r) ** 2 + (y - c) ** 2) + max(0.4, slope_err)  #

    res = minimize(distance_fn, [x0, y0, z0], method='Powell')  # 第一个参数是目标函数，第二个参数是初始化的值。
    x_new, y_new, z_new = res.x
    return x_new, y_new, z_new


# 清除在世界坐标系上距离相邻的点（选择置信度大的那一个，就是非极大值抑制）
def clear_duplicates(coords):
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2) ** 2).sum())
            if distance < DISTANCE_THRESH_CLEAR:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]


def extract_coords(prediction, flipped=False, threshold=0.6):
    # 从预测输出的张量（或者说是特征图图）中找到置信度大于0的位置，然后从中取出除回归的量，并且清楚重复预测的位置。返回一个列表，该列表中含有数量等同于预测出的车辆的个数的字典，字典中是七个回归预测量。
    logits = prediction[0]  # 第一位是对于置信度的预测。这是在单独的一个batch中预测出来的。最终的格式是CHW。
    # logits = torch.max_pool1d(torch.tensor(logits), kernel_size=3)  # 对置信度池化筛选。
    # logits = logits[0].numpy()  # 取第一维
    regr_output = prediction[1:]  # 后面7位是对回归量的预测。
    logits = np.where(logits < - 10, -10, logits)  # exp的值呈现指数级的增长，很容易就溢出了。所以要把小于-10统统置为-10
    points = np.argwhere(
        (1 / (1 + np.exp(-logits))) > threshold)  # 预测的置信度在未通过sigmoid层时取得点，因为高斯标签，大量的点被赋予置信度，所以应该筛选掉这些低置信度的点。
    col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])  # 对其进行排序
    coords = []
    for r, c in points:  # r,c 是预测得到的置信度大于零的点坐标。（也就是车的中心位置在像素坐标系下的坐标）；如果该图中没有置信度大于零的点，返回的列表为空。
        regr_dict = dict(zip(col_names, regr_output[:, r, c]))  # 从预测的值当中，得到相应的字典。
        coords.append(_regr_back(regr_dict))  # 将预测得到的值转换成原始标签的形式
        coords[-1]['confidence'] = (1 / (1 + np.exp(-logits[r, c])))
        coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] = \
            optimize_xy(r, c,
                        coords[-1]['x'],
                        coords[-1]['y'],
                        coords[-1]['z'],
                        flipped)  # 由置信度大于零点可以得到其相应的其余7个属性的值，由其x,y,z属性计算其在像素坐标系下坐标的点，然后对其和其预测的点进行误差最小的优化，优化得到的参数新的x,y,z点的坐标
    coords = clear_duplicates(coords)  # 清除距离相近的点，进行非极大值抑制
    return coords  # 返回包含有数量不等（一张图片中车的数目）的字典的列表。


def coords2str(coords, names=('yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence')):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)


def threshold_s(s, confidence_threshold):
    """
     Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    """

    coords = []
    if isinstance(s, str) and (s != ''):
        for l in np.array(s.split()).reshape([-1, 7]):
            if l.astype('float')[6] >= confidence_threshold:
                tmp = ' '.join(l)
                coords.append(tmp)
        return ' '.join(coords)
    else:
        return ''
