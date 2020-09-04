# 01.Task


1. CompetitionChallenge

Develop an algorithm to estimate the absolute pose of vehicles (6 degrees of    freedom) from a single image in a real-world traffic environment.


1. **Dataset &**Submission Requirements

    * The dataset contains 4k images of streets in train, taken from the roof of a car.E.g.

![图片](https://uploader.shimo.im/f/z6Fjqu5ncKvAQkc7.png!thumbnail)

The label file of the train dataset is as follows:

![图片](https://uploader.shimo.im/f/kTGmAH6HViD1uUvq.png!thumbnail)


    * We're attempting to predict the position and orientation of all un-masked cars in the test images. we should also provide a confidence score indicating how sure we are of your prediction.
# 02.Our Model**& Works**


1. Network

Firstly, we use**4****double convolution + max pooling**(**dm**)  blocksto do the preliminary extraction of semantic information.


    * **double convolution + max pooling:**As the channel(**C**) increases, the feature layer size(**W, H**) is halved.

Then we use**EfficientNet**for further extraction.The**output**of this step is**upsampled**and**connecte**on the**fourth**tensor of**dm**block result(similar toshortcut).The connected part is**upsampled**again, and then**connecte**with the**third****dm**block result(shortcut again).

Finally, a**conv2d**layer is used to adjust the**channel**to**n_calsses(**'confidence', 'yaw','pitch_sin','pitch_cos','roll','x','y','z',**)**for output.


1. Tricks

    1. **Data Augmentation**

        * Image cut
        * random flip
        * rgb shift
        * larger image size
    2. **Pitch****Split(Labels)**
        * pitch_sin
        * pitch_cos
    3. **Gaussian masks**

![图片](https://uploader.shimo.im/f/gOsCmJdLuKYP8M8u.png!thumbnail)

    4. **Gaussian Focal Loss**

        * formula：

![图片](https://uploader.shimo.im/f/0JBV7WX4aQwPjowH.png!thumbnail)


1. mAp

The**key**is the matching of GT and predicted value.


    1. **Sort**the multiple**confidences**in the prediction result of a picture, and then loop judgment from the largest. 对图片的预测结果中的多个置信度进行排序，然后根据最大的判断循环。
    2. The 6 sets of attitude data under a certain confidence are**matched**with the GT and the predicted value according to the minimum distance of x, y, z. The angular distance is also calculated at this time. 在某个置信度下的6组姿态数据，按x,y,z距离最小来匹配GT和预测值。这时也计算了角度距离。
    3. If all meet the**thresholds**(thre_tr_dist, thre_ro_dist), delete the corresponding values in GT, and add 1 (**TP**) to result_flg. If the thresholds are not met, add 0 (**FP**) to result_flg directly.如果都满足阈值, 将GT中对应的这些值删掉,同时在result_flg添加1（TP）.如果不满足阈值，则直接在result_flg添加0（FP）。
    4. The total number of vehicles in the GT (**TP+FN**).GT中的总车辆数(TP+FN)。

1. Result

We adjusted the baseline with an initial mAP of 0.026 to**0.070**.














