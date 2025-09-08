import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from ultralytics.models import YOLO 
import os
from ultralytics.utils.loss import v8ClassificationLoss  # 导入 v8ClassificationLoss
from ultralytics.utils.loss import VarifocalLoss

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="__floordiv__ is deprecated")



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 定义数据增强函数
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.3),
    A.CoarseDropout(max_holes=8, max_height=10, max_width=10, min_holes=1, min_height=5, min_width=5, p=0.5),  # 使用 CoarseDropout 替代 Cutout
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
    A.Blur(blur_limit=3, p=0.2),
    A.OpticalDistortion(p=0.3),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def load_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image at path {img_path} could not be loaded.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
    augmented = transform(image=image)
    return augmented['image']

if __name__ == '__main__':
    # 加载 YOLO 模型
    model = YOLO(model='./v8/yolov8.yaml')
    
    # 实例化损失函数
    # loss_fn = v8ClassificationLoss()
    # loss_fn = VarifocalLoss()

    # 使用模型进行训练，并应用内置和自定义数据增强
    model.train(
        data='./data.yaml',
        epochs=100,
        # conf=0.6,
        batch=8,
        device='0',
        imgsz=640,
        workers=16,
        cache=False,
        amp=True,
        mosaic=True,  # 启用 Mosaic 数据增强
        hsv_h=0.015,  # 色调
        hsv_s=0.7,  # 饱和度
        hsv_v=0.4,  # 亮度
        flipud=0.0,  # 垂直翻转
        fliplr=0.5,  # 水平翻转 
        project='/home/neaucs2/usr/lsw2/YOLO-20241007/runs/v8',
        name='CG'
    )
