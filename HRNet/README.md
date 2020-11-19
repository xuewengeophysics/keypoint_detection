# Keypoint Detection by HRNet   

`original code`
clone from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

参考https://github.com/lxy5513/hrnet



## Demo

```shell
python tools/human_keypoint_inference.py
```

 

## Model Download 
+ 下载pose_hrnet_*.pth模型文件，保存到models/pytorch/pose_coco文件夹中
  + address: https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA)   

+ 下载yolov3目标检测模型文件，保存到/lib/detector/yolo文件夹中
  + yolov3 model download: wget https://pjreddie.com/media/files/yolov3.weights



## Main Steps

1. 人体目标检测：

   ```python
   bboxs, scores = yolo_det(args.img_input, human_model, confidence=0.5)  # bboxes (N, 4) [x0, y0, x1, y1]
   ```

   

2. 根据上一步得到的bbox提取单个的人体图像：

   ```
   inputs, origin_img, center, scale = preprocess(args.img_input, bboxs, scores, cfg)
   ```

   

3. 关键点检测，得到每个关键点的heatmap：

   ```
   output = model(inputs)
   ```

   

4. heatmap后处理，得到关键点坐标：

   ```
   preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))
   ```

   

## ONNX Model Inference

1. pth模型转onnx：

   ```python
   python tools/pytorch_model2onnx.py --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
   --pth models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth
   ```

   

2. 根据上一步得到的*.onnx模型进行推理：

   ```
   python tools/human_keypoint_inference_onnx.py
   ```

   