from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")

    model.train(
        data="config.yaml",
        epochs=1000,
        imgsz=640,
        device=0,
        workers=2,   # IMPORTANT for Windows
        patience=0   # 🔥 disable EarlyStopping completely
    )
