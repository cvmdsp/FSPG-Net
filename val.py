import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('./best.pt')
    model.val(data='./mydata.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True,
              # save_txt=True,
              conf=0.25,
              iou=0.5
              )