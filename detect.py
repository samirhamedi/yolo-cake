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
        print("ERROR: Model not found.")
        sys.exit(0)

    model = YOLO(model_path)
    labels = model.names

    img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
    vid_ext_list = ['.avi', '.mp4', '.mov', '.mkv', '.wmv']

    # Determine source type
    if os.path.isdir(img_source):
        source_type = 'folder'
    elif os.path.isfile(img_source):
        _, ext = os.path.splitext(img_source)
        if ext.lower() in img_ext_list:
            source_type = 'image'
        elif ext.lower() in vid_ext_list:
            source_type = 'video'
        else:
            print(f"Unsupported file type: {ext}")
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
    if record and source_type not in ['video', 'usb']:
        print("Recording only works for video or webcam.")
        sys.exit(0)

    # Load source
    cap = None
    imgs_list = []

    if source_type == "image":
        imgs_list = [img_source]

    elif source_type == "folder":
        imgs_list = [
            f for f in glob.glob(img_source + "/*")
            if os.path.splitext(f)[1].lower() in img_ext_list
        ]

    elif source_type == "video":
        cap = cv2.VideoCapture(img_source)  # FIXED: no CAP_DSHOW for video

    elif source_type == "usb":
        cap = cv2.VideoCapture(usb_idx, cv2.CAP_DSHOW)  # webcam only

        if resize:
            cap.set(3, resW)
            cap.set(4, resH)

    # Set up recorder now that the source dimensions are known
    if record:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        if resize:
            out_w, out_h = resW, resH
        else:
            out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Keep the same container/format as the source file
        if source_type == "video":
            base, out_ext = os.path.splitext(os.path.basename(img_source))
        else:
            base, out_ext = "webcam", ".mp4"
        out_ext = out_ext.lower()

        # Pick a codec that matches the container
        codec_map = {
            ".mp4": "mp4v",
            ".mov": "mp4v",
            ".mkv": "mp4v",
            ".avi": "XVID",
            ".wmv": "WMV2",
        }
        fourcc_str = codec_map.get(out_ext, "mp4v")

        out_path = f"{base}_detected{out_ext}"
        recorder = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*fourcc_str),
            src_fps,
            (out_w, out_h),
        )
        print(f"Recording to {out_path} "
              f"({out_w}x{out_h} @ {src_fps:.2f} fps, codec {fourcc_str})")

    # Colors
    bbox_colors = [
        (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133),
        (88, 159, 106), (96, 202, 231), (159, 124, 168), (169, 162, 241),
        (98, 118, 150), (172, 176, 184)
    ]

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

        elif source_type in ["video", "usb"]:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Reached end of video or camera disconnected.")
                break

        if resize:
            frame = cv2.resize(frame, (resW, resH))

        # YOLO detection
        results = model.predict(frame, verbose=False)
        detections = results[0].boxes

        object_count = 0

        for det in detections:
            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy

            classidx = int(det.cls.item())
            classname = labels[classidx]
            conf = det.conf.item()

            if conf >= min_thresh:
                color = bbox_colors[classidx % len(bbox_colors)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                label = f"{classname}: {int(conf * 100)}%"
                cv2.putText(frame, label, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                object_count += 1

        # FPS
        if source_type in ["video", "usb"]:
            cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"Objects: {object_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("YOLO Detection", frame)

        if record and recorder:
            recorder.write(frame)

        # Key handling
        if source_type in ["image", "folder"]:
            key = cv2.waitKey(0)
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

    print(f"Average pipeline FPS: {avg_fps:.2f}")

    if cap:
        cap.release()
    if recorder:
        recorder.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
