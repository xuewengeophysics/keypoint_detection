# Human Pose Estimation using Keypoint RCNN in PyTorch笔记

+ Blog: [Human Pose Estimation using Keypoint RCNN in PyTorch](https://learnopencv.com/human-pose-estimation-using-keypoint-rcnn-in-pytorch)
+ Code: [spmallick/learnopencv/PyTorch-Keypoint-RCNN](spmallick/learnopencv/PyTorch-Keypoint-RCNN)

## Table of Contents

1. Evolution of Keypoint RCNN Architecture
2. Input-Output Format
4. Loss Function in Keypoint-RCNN
7. Evaluation Metric in Keypoint Detection

## Evolution of Keypoint RCNN Architecture

1. `N` is the number of objects proposed by the Region-Proposal Layer.
2. `C` is `2`, the [MS-COCO Dataset](https://cocodataset.org/#home) offers keypoints only for the person class.
3. `K` is the number of keypoints per person, which is `17`.

![img](https://learnopencv.com/wp-content/uploads/2021/05/KeypointRCNN.jpg)

+ The output from Keypoint-RCNN is now sized `[N, K=17, 56, 56]`. Each of the K channels corresponds to a specific keypoint (for eg: left-elbow, right-ankle etc).
+ The final class-scores will be of size `[N, 2]`:
  + one for background
  + the other for the person class
+ The box-predictions will be sized `[N, 2 * 4]`.

![Shows a table depicting the different keypoints in COCO dataset. Also has an example image with a skeletal structure of a human.](https://learnopencv.com/wp-content/uploads/2021/05/fix-overlay-issue.jpg)

## Input Output Format

Input to the model a tensor of size `[batch_size, 3, height, width]`. Note that the original image should be normalized (i.e. the pixel values in the image should range between 0 and 1).

```python
output = model([img_tensor])[0]
```

The variable `output` is a dictionary, with the following keys and values:

+ ***boxes*** – A tensor of size `[N, 4]`, where `N` is the number of objects detected.

+ ***labels*** – A tensor of size `[N]`, depicting the class of the object. 
  + This is always 1 because each detected box belongs to a person.
  +  0 stands for the background class.
+ ***scores*** – A tensor of size `[N]`, depicting the confidence score of the detected object.

+ ***keypoints*** – A tensor of size` [N, 17, 3]`, depicting the 17 joint locations of `N` number of persons. Out of 3, the first two numbers are the coordinates `x` and `y`, and the third one depicts the visibility.
  + 0, when keypoint is invisible
  + 1, when keypoint is visible

+ ***keypoints_scores*** – A tensor of size `[N, 17]`, depicting the score for all the keypoints, for each detected person.

## Loss Function in Keypoint-RCNN

![\[\frac{-\sum_{h,w}\left[Y_{k,h,w}==1 \right ] \left(Y_{k,h,w} * \log \left(\text{softmax}\left(\widehat{Y}_{k,h,w} \right ) \right )\right )}{\sum_{h,w} \left[Y_{k,h,w} == 1 \right ]}\]](https://learnopencv.com/wp-content/ql-cache/quicklatex.com-fa0b20ef4a8f4070e6f909b15173325c_l3.png)

## Evaluation Metric in Keypoint Detection

+ [Object Keypoint Similarity (OKS)](https://cocodataset.org/#keypoints-eval)

![OKS](https://learnopencv.com/wp-content/ql-cache/quicklatex.com-ec2c07ae0c77b371b7ff2c1dc4b77f49_l3.png)

Where `d_i` is the Euclidean distance between predicted and ground-truth, `s` is the object’s scale, and `k_i` is a constant for a specific keypoint.

### How to fix the values for `k`?

Well, as we mentioned earlier, k is a constant factor for each keypoint and it remains the same for all samples. It turns out that k is a measure of standard-deviation of a particular keypoint. Essentially, the value of `k` for keypoints on the face (eyes, ears, nose) have a relatively smaller standard deviation than the keypoints on the body (hips, shoulders, knees).



更新于2021-06-22