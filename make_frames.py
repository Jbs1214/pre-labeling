#!/usr/bin/env python3
"""영상 -> 프레임 추출 + 자동 라벨(사전 라벨).

실행:
    python make_frames.py A.mp4 B.mp4 --every 5 --model ./weights/yolo11n.pt
    python make_frames.py A.mp4 B.mp4 --every 5 --model ./weights/yolo11n.pt --slice 640   # SAHI 사용

    dataset/images/{영상이름}_{프레임번호}.jpg   --every 초마다 한 장
    dataset/labels/{같은 이름}.txt               YOLO 형식 자동 라벨 (labeler.py 의 Detector 그대로 사용)

이미 있는 이미지·라벨은 건너뛴다(검수한 라벨을 덮어쓰지 않는다).
자동 라벨은 출발점일 뿐이다. labeler.py 로 검수해 '확인 완료'로 바꾼 것만 학습에 쓴다.
"""
import argparse
import sys
from pathlib import Path

import cv2

from labeler import COCO_TO_OURS, Detector, atomic_write


class SahiDetector:
    """SAHI: 프레임을 slice x slice 조각으로 잘라 검출하고 합친다(멀리 있는 작은 객체용). Detector 와 같은 출력."""

    def __init__(self, model_path: Path, slice_size: int, conf: float, device, mapping: str):
        from sahi import AutoDetectionModel
        self.model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(model_path),
                                                        confidence_threshold=conf, device=device or "cpu")
        self.slice_size, self.mapping = slice_size, mapping

    def predict(self, image_path: Path, conf: float):
        from sahi.predict import get_sliced_prediction
        r = get_sliced_prediction(str(image_path), self.model, slice_height=self.slice_size, slice_width=self.slice_size,
                                  overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0)
        W, H = r.image_width, r.image_height
        out = []
        for p in r.object_prediction_list:
            cid = COCO_TO_OURS.get(p.category.id) if self.mapping == "coco" else p.category.id
            if cid is None:
                continue
            x1, y1, x2, y2 = p.bbox.to_xyxy()
            cx, cy, w, h = (min(max(v, 0.0), 1.0) for v in ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
            out.append({"c": cid, "cx": cx, "cy": cy, "w": w, "h": h})
        return out


def extract(video: Path, out: Path, every: float) -> list[Path]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps * every))
    saved = []
    for idx in range(0, n, step):
        p = out / f"{video.stem}_{idx:06d}.jpg"
        if not p.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            if not ok:
                print(f"  {idx}번 프레임 읽기 실패, 건너뜀", file=sys.stderr)
                continue
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(p)
    cap.release()
    print(f"{video.name}: {fps:.2f}fps, {n}프레임 -> {step}프레임마다 {len(saved)}장")
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--every", type=float, default=5.0, help="몇 초마다 한 장 뽑을지")
    ap.add_argument("--model", type=Path, default=None, help="YOLO 가중치 파일 (없으면 추출만)")
    ap.add_argument("--map", choices=["coco", "ours"], default="coco")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    ap.add_argument("--slice", type=int, default=0, help="SAHI 조각 크기(예: 640). 0이면 SAHI 없이 통째로 검출")
    args = ap.parse_args()

    images, labels = args.data / "images", args.data / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    frames = [p for v in args.videos for p in extract(v, images, args.every)]

    if args.model is None:
        return 0
    if not args.model.is_file():
        print(f"모델 파일을 찾을 수 없습니다: {args.model}", file=sys.stderr)
        return 1
    if args.slice:
        det = SahiDetector(args.model, args.slice, args.conf, args.device, args.map)
    else:
        det = Detector(args.model, args.imgsz, args.device, args.map)
    todo = [p for p in frames if not (labels / (p.stem + ".txt")).exists()]
    for i, p in enumerate(todo, 1):
        boxes = det.predict(p, args.conf)
        text = "".join(f"{b['c']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}\n" for b in boxes)
        atomic_write(labels / (p.stem + ".txt"), text.encode("utf-8"))
        if i % 25 == 0 or i == len(todo):
            print(f"자동 라벨 {i}/{len(todo)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
