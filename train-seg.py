from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n-seg.pt")

    model.train(
        data="config.yaml",
        epochs=300,
        imgsz=640,
        device=0,
        workers=2   # IMPORTANT for Windows
    )
