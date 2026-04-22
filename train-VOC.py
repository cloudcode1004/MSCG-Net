from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", message="adaptive_max_pool2d_backward_cuda does not have a deterministic implementation")

if __name__ == '__main__':
    model = YOLO('/ultralytics/cfg/models/11/MSCG-Net.yaml')
    model.train(data='/ultralytics/cfg/datasets/Visdrone.yaml',
                imgsz=(1024, 640),
                rect=True,
                batch=16, device=0, epochs=400,
                )
