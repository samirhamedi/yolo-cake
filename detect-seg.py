import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--resolution', default=None)
    parser.add_argument('--record', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.model
    img_source = args.source
    min_thresh = args.thresh
    user_res = args.resolution
    record = args.record

    if not os.path.exists(model_path):
        print("Model not found.")
        sys.exit(0)

    model = YOLO(model_path)
    labels = model.names

    img_ext = ['.jpg', '.jpeg', '.png', '.bmp']
    vid_ext = ['.avi', '.mp4', '.mov', '.mkv', '.wmv']

    # Determine source type
    if os.path.isdir(img_source):
        source_type = 'folder'
    elif os.path.isfile(img_source):
        _, ext = os.path.splitext(img_source)
        if ext.lower() in img_ext:
            source_type = 'image'
        elif ext.lower() in vid_ext:
            source_type = 'video'
        else:
            print("Unsupported file type.")
            sys.exit(0)
    elif "usb" in img_source:
        source_type = 'usb'
        usb_idx = int(img_source[3:])
    else:
        print("Invalid source.")
        sys.exit(0)

    # Resolution
    resize = False
    if user_res:
        resize = True
        resW, resH = map(int, user_res.split("x"))

    # Recording
    recorder = None
    if record:
        if source_type not in ['video', 'usb']:
            print("Recording only works for video or webcam.")
            sys.exit(0)
        if not user_res:
            print("Specify resolution when recording.")
            sys.exit(0)

        recorder = cv2.VideoWriter(
            "demo1.avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            30,
            (resW, resH)
        )

    # Load source
    cap = None
    imgs_list = []

    if source_type == "image":
        imgs_list = [img_source]

    elif source_type == "folder":
        imgs_list = [
            f for f in glob.glob(img_source + "/*")
            if os.path.splitext(f)[1].lower() in img_ext
        ]

    elif source_type in ["video", "usb"]:
        cap_arg = img_source if source_type == "video" else usb_idx
        cap = cv2.VideoCapture(cap_arg, cv2.CAP_DSHOW)

        if resize:
            cap.set(3, resW)
            cap.set(4, resH)

    avg_fps = 0
    fps_buffer = []
    fps_len = 200
    img_count = 0

    while True:
        t_start = time.perf_counter()

        # Load frame
        if source_type in ["image", "folder"]:
            if img_count >= len(imgs_list):
                print("All images processed.")
                break
            frame = cv2.imread(imgs_list[img_count])
            img_count += 1

        elif source_type == "video":
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

        elif source_type == "usb":
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Camera disconnected.")
                break

        if resize:
            frame = cv2.resize(frame, (resW, resH))

        # Run segmentation
        results = model.predict(frame, verbose=False)
        r = results[0]

        object_count = 0

        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            boxes = r.boxes

            for i, mask in enumerate(masks):
                conf = boxes[i].conf.item()
                classidx = int(boxes[i].cls.item())

                if conf < min_thresh:
                    continue

                object_count += 1

                # Resize mask to frame size
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_bin = (mask_resized > 0.5).astype(np.uint8)

                overlay = np.zeros_like(frame)
                overlay[:, :, 1] = mask_bin * 255  # green mask

                frame = cv2.addWeighted(frame, 1.0, overlay, 0.4, 0)

                classname = labels[classidx]
                label = f"{classname}: {int(conf * 100)}%"
                cv2.putText(frame, label, (10, 30 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # FPS
        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"Objects: {object_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("YOLO Segmentation", frame)

        if record and recorder:
            recorder.write(frame)

        # KEY HANDLING (THIS FIXES YOUR WINDOW CLOSING)
        if source_type in ["image", "folder"]:
            key = cv2.waitKey(0)  # WAIT FOREVER
        else:
            key = cv2.waitKey(5)

        if key in [ord("q"), ord("Q")]:
            break

        # FPS calc
        t_stop = time.perf_counter()
        fps = 1.0 / (t_stop - t_start)

        if len(fps_buffer) >= fps_len:
            fps_buffer.pop(0)
        fps_buffer.append(fps)

        avg_fps = np.mean(fps_buffer)

    print(f"Average FPS: {avg_fps:.2f}")

    if cap:
        cap.release()
    if recorder:
        recorder.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
