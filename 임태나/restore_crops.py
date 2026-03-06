import shutil
from pathlib import Path

DATASET = Path(r"C:\gachikium\dataset")

# v1에서 이동된 불량 폴더(원본 crop들이 여기로 갔을 가능성이 큼)
BAD_DIR = DATASET / "crop_bad"

# 원복 대상
TRAIN_CROP = DATASET / "train_crop"
TEST_CROP  = DATASET / "test_crop"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"}

def list_images(d: Path):
    if not d.exists():
        return []
    return [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def restore_split(split_name: str, dst_root: Path):
    src_root = BAD_DIR / split_name
    if not src_root.exists():
        print(f"[SKIP] {src_root} 없음")
        return

    dst_root.mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in src_root.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"[SKIP] {src_root} 안에 클래스 폴더가 없음")
        return

    for cls_dir in class_dirs:
        dst_cls = dst_root / cls_dir.name
        dst_cls.mkdir(parents=True, exist_ok=True)

        imgs = list_images(cls_dir)
        for p in imgs:
            dst = dst_cls / p.name
            # 같은 이름 있으면 덮어씀(원복 목적)
            shutil.copy2(p, dst)

        print(f"[OK] {split_name}/{cls_dir.name}: copied {len(imgs)}")

print("=== RESTORE from crop_bad -> train_crop/test_crop (COPY ONLY) ===")
restore_split("train", TRAIN_CROP)
restore_split("test", TEST_CROP)

print("\nDONE.")
print("train_crop:", TRAIN_CROP)
print("test_crop :", TEST_CROP)