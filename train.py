import warnings, os
from ultralytics import YOLO
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    model = YOLO('./ours.yaml', task='detect'
        )
    model.train(data='./mydata.yaml',
                imgsz=640,
                epochs=300,
                batch=4,
                workers=2,
                # device='0,1',
                # patience=0,
                # resume=True,
                amp=False,
                # fraction=0.2,
                )

