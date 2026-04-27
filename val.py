from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('weights/best.pt')
    model.val(data='/ultralytics/cfg/datasets/VisDrone.yaml',
              imgsz=(1024, 640),
              rect=True,
              device=0, split='val', save_json=True, batch=16,
              )

