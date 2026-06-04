import os

from ultralytics.models import YOLO
from ultralytics.nn.modules.conv import CBAM

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == '__main__':
    model = YOLO(model='/home/neaucs2/usr/ultralytics/cfg/models/v8/yolov8.yaml')
   
    model.train(
        data='./data.yaml', epochs=100, batch=8, device='0', imgsz=640, workers=16, 
        cache=False, amp=True, 
        project='/home/neaucs2/usr/lsw2/YOLO-20241007/runs/shiyan',
        name='YOLOCG'
    )
