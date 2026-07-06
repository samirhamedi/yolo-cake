"""Verify the corrected model against every labeled image in the dataset.

For each image it compares the model's predicted class-index distribution to
the ground-truth label file, then reports per-split and overall accuracy.
"""
import glob
import os
from collections import Counter

from ultralytics import YOLO

MODEL = "history/best-fixed.pt"
CONF = 0.25
SPLITS = ["train", "val"]

model = YOLO(MODEL)
names = model.names


def gt_counts(label_path):
    if not os.path.exists(label_path):
        return Counter()
    with open(label_path) as f:
        return Counter(int(line.split()[0]) for line in f if line.strip())


overall_img_ok = 0
overall_img_total = 0
overall_obj_tp = 0
overall_obj_gt = 0

for split in SPLITS:
    img_dir = f"dataset/{split}/images"
    lbl_dir = f"dataset/{split}/labels"
    images = sorted(glob.glob(f"{img_dir}/*.jpg"))

    print(f"\n=== {split.upper()} ({len(images)} images) ===")
    split_img_ok = 0
    split_obj_tp = 0
    split_obj_gt = 0

    for img in images:
        stem = os.path.splitext(os.path.basename(img))[0]
        gt = gt_counts(f"{lbl_dir}/{stem}.txt")

        res = model.predict(img, verbose=False, conf=CONF)[0]
        pred = Counter(int(b.cls.item()) for b in res.boxes)

        # object-level matches: min(pred, gt) per class
        tp = sum(min(gt[c], pred[c]) for c in gt)
        total_gt = sum(gt.values())
        exact = pred == gt

        split_obj_tp += tp
        split_obj_gt += total_gt
        split_img_ok += int(exact)

        if not exact:
            gt_named = {names[c]: n for c, n in sorted(gt.items())}
            pred_named = {names[c]: n for c, n in sorted(pred.items())}
            print(f"  [MISMATCH] {stem}")
            print(f"      GT  : {gt_named}")
            print(f"      PRED: {pred_named}")

    print(f"  Exact-match images : {split_img_ok}/{len(images)}")
    print(f"  Object recall      : {split_obj_tp}/{split_obj_gt}"
          f" ({(split_obj_tp / split_obj_gt * 100 if split_obj_gt else 0):.1f}%)")

    overall_img_ok += split_img_ok
    overall_img_total += len(images)
    overall_obj_tp += split_obj_tp
    overall_obj_gt += split_obj_gt

print("\n=== OVERALL ===")
print(f"  Exact-match images : {overall_img_ok}/{overall_img_total}"
      f" ({overall_img_ok / overall_img_total * 100:.1f}%)")
print(f"  Object recall      : {overall_obj_tp}/{overall_obj_gt}"
      f" ({overall_obj_tp / overall_obj_gt * 100:.1f}%)")
