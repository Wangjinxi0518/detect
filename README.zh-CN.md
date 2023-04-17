<details open>
<summary>安装</summary>

在一个 [**Python>=3.7**](https://www.python.org/) 环境中，使用 [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/)，通过 pip 安装 ultralytics 软件包以及所有[依赖项](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt)。

```bash
pip install ultralytics
```

</details>

<details open>
<summary>Usage</summary>

#### CLI

YOLOv8 可以在命令行界面（CLI）中直接使用，只需输入 `yolo` 命令：

```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

`yolo` 可用于各种任务和模式，并接受其他参数，例如 `imgsz=640`。查看 YOLOv8 [CLI 文档](https://docs.ultralytics.com/usage/cli)以获取示例。

#### Python

YOLOv8 也可以在 Python 环境中直接使用，并接受与上述 CLI 示例中相同的[参数](https://docs.ultralytics.com/usage/cfg/)：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("yolov8n.pt")  # 加载预训练模型（建议用于训练）

# 使用模型
model.train(data="coco128.yaml", epochs=3)  # 训练模型
metrics = model.val()  # 在验证集上评估模型性能
results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
```

[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models) 会自动从最新的 Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)中下载。查看 YOLOv8 [Python 文档](https://docs.ultralytics.com/usage/python)以获取更多示例。

</details>

## <div align="center">模型</div>

所有的 YOLOv8 预训练模型都可以在此找到。检测、分割和姿态模型在 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/datasets/coco.yaml) 数据集上进行预训练，而分类模型在 [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/datasets/ImageNet.yaml) 数据集上进行预训练。

在首次使用时，[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models) 会自动从最新的 Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)中下载。

<details open><summary>检测</summary>

查看 [检测文档](https://docs.ultralytics.com/tasks/detect/) 以获取使用这些模型的示例。

| 模型                                                                                 | 尺寸<br><sup>(像素) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | ------------------- | -------------------- | ----------------------------- | ---------------------------------- | ---------------- | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                 | 37.3                 | 80.4                          | 0.99                               | 3.2              | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                 | 44.9                 | 128.4                         | 1.20                               | 11.2             | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                 | 50.2                 | 234.7                         | 1.83                               | 25.9             | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                 | 52.9                 | 375.2                         | 2.39                               | 43.7             | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                 | 53.9                 | 479.1                         | 3.53                               | 68.2             | 257.8             |

- **mAP<sup>val</sup>** 值是基于单模型单尺度在 [COCO val2017](http://cocodataset.org) 数据集上的结果。
  <br>通过 `yolo val detect data=coco.yaml device=0` 复现
- **速度** 是使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例对 COCO val 图像进行平均计算的。
  <br>通过 `yolo val detect data=coco128.yaml batch=1 device=0|cpu` 复现

</details>

<details><summary>分割</summary>

查看 [分割文档](https://docs.ultralytics.com/tasks/segment/) 以获取使用这些模型的示例。

| 模型                                                                                         | 尺寸<br><sup>(像素) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | ------------------- | -------------------- | --------------------- | ----------------------------- | ---------------------------------- | ---------------- | ----------------- |
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                 | 36.7                 | 30.5                  | 96.1                          | 1.21                               | 3.4              | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                 | 44.6                 | 36.8                  | 155.7                         | 1.47                               | 11.8             | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                 | 49.9                 | 40.8                  | 317.0                         | 2.18                               | 27.3             | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                 | 52.3                 | 42.6                  | 572.4                         | 2.79                               | 46.0             | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                 | 53.4                 | 43.4                  | 712.1                         | 4.02                               | 71.8             | 344.1             |

- **mAP<sup>val</sup>** 值是基于单模型单尺度在 [COCO val2017](http://cocodataset.org) 数据集上的结果。
  <br>通过 `yolo val segment data=coco.yaml device=0` 复现
- **速度** 是使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例对 COCO val 图像进行平均计算的。
  <br>通过 `yolo val segment data=coco128-seg.yaml batch=1 device=0|cpu` 复现

</details>

<details><summary>分类</summary>

查看 [分类文档](https://docs.ultralytics.com/tasks/classify/) 以获取使用这些模型的示例。

| 模型                                                                                         | 尺寸<br><sup>(像素) | acc<br><sup>top1 | acc<br><sup>top5 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
| -------------------------------------------------------------------------------------------- | ------------------- | ---------------- | ---------------- | ----------------------------- | ---------------------------------- | ---------------- | ------------------------ |
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                 | 66.6             | 87.0             | 12.9                          | 0.31                               | 2.7              | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                 | 72.3             | 91.1             | 23.4                          | 0.35                               | 6.4              | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                 | 76.4             | 93.2             | 85.4                          | 0.62                               | 17.0             | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                 | 78.0             | 94.1             | 163.0                         | 0.87                               | 37.5             | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                 | 78.4             | 94.3             | 232.0                         | 1.01                               | 57.4             | 154.8                    |

- **acc** 值是模型在 [ImageNet](https://www.image-net.org/) 数据集验证集上的准确率。
  <br>通过 `yolo val classify data=path/to/ImageNet device=0` 复现
- **速度** 是使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例对 ImageNet val 图像进行平均计算的。
  <br>通过 `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu` 复现

</details>

<details><summary>姿态</summary>

查看 [姿态文档](https://docs.ultralytics.com/tasks/) 以获取使用这些模型的示例。

| 模型                                                                                                 | 尺寸<br><sup>(像素) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------------------------------------------------------------------------------------------- | ------------------- | --------------------- | ------------------ | ----------------------------- | ---------------------------------- | ---------------- | ----------------- |
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                 | 49.7                  | 79.7               | 131.8                         | 1.18                               | 3.3              | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                 | 59.2                  | 85.8               | 233.2                         | 1.42                               | 11.6             | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                 | 63.6                  | 88.8               | 456.3                         | 2.00                               | 26.4             | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                 | 67.0                  | 89.9               | 784.5                         | 2.59                               | 44.4             | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                 | 68.9                  | 90.4               | 1607.1                        | 3.73                               | 69.4             | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                | 71.5                  | 91.3               | 4088.7                        | 10.04                              | 99.1             | 1066.4            |

- **mAP<sup>val</sup>** 值是基于单模型单尺度在 [COCO Keypoints val2017](http://cocodataset.org) 数据集上的结果。
  <br>通过 `yolo val pose data=coco-pose.yaml device=0` 复现
- **速度** 是使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例对 COCO val 图像进行平均计算的。
  <br>通过 `yolo val pose data=coco8-pose.yaml batch=1 device=0|cpu` 复现

</details>