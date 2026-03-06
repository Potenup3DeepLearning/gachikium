import cv2, numpy as np, random, hashlib, shutil
from pathlib import Path
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

# ======================
# 0) 경로 설정 (너 그대로)
# ======================
TRAIN_COPY_DIR = Path(r"C:\gachikium\dataset\train_copy")   # 클래스당 80
TRAIN_DIR      = Path(r"C:\gachikium\dataset\train")        # 클래스당 70
TEST_DIR       = Path(r"C:\gachikium\dataset\test")         # 있으면 베스트

TRAIN_CROP_DIR = Path(r"C:\gachikium\dataset\train_crop")   # 현재 크롭 결과(56)
TEST_CROP_DIR  = Path(r"C:\gachikium\dataset\test_crop")    # 현재 크롭 결과(14)

TRAIN_OUT_DIR  = Path(r"C:\gachikium\dataset\train_crop_fixed_v2")
TEST_OUT_DIR   = Path(r"C:\gachikium\dataset\test_crop_fixed_v2")

BAD_DIR        = Path(r"C:\gachikium\dataset\crop_bad_v2")   # 불량 crop "복사" 보관
FAIL_DIR       = Path(r"C:\gachikium\dataset\crop_fail_v2")  # 보충 실패 원본 "복사" 보관

TRAIN_TARGET = 56
TEST_TARGET  = 14
OUT_SIZE = 224
SEED = 42

# === "중요": 기존 crop을 건드리지 않게 (move 금지)
BAD_ACTION = "copy"   # "copy" or "move" (권장: copy)
FAIL_ACTION = "copy"

# === crop 검수 강도(너 상황에 맞게 완화)
# crop 이미지 자체에서 얼굴 재검출은 OFF(너가 말한 ‘정상도 bad로 가는’ 문제 해결)
USE_FACE_RECHECK_ON_CROP = False

# "진짜 이상한 것만" 걸러내는 품질 기준 (완화)
DARK_THRESH = 8
VAR_THRESH  = 12

# 보충할 때 원본에서 얼굴이 너무 작은 사진은 버림(전신샷 방지)
MIN_FACE_RATIO_IN_ORIGINAL = 0.12  # 얼굴 bbox 면적 / 이미지 면적 최소 비율

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
random.seed(SEED)
np.random.seed(SEED)

# ======================
# 1) 유틸
# ======================
def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_unicode_jpg(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(str(path))

def clear_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def list_images(folder: Path):
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def img_hash(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()

def file_hash(path: Path, chunk=1024*1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def is_bad_crop_simple(img, dark_thresh=DARK_THRESH, var_thresh=VAR_THRESH):
    """
    crop 검수는 '가볍게':
    - 너무 어두움(검은 막대/암부)
    - 너무 단조로움(블러/막대/배경만)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.mean() < dark_thresh:
        return True
    if gray.var() < var_thresh:
        return True
    return False

def safe_put(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

# ======================
# 2) 얼굴 검출기
# ======================
app = FaceAnalysis(name="buffalo_l")
# 더 잘 잡히게 약간 완화
app.prepare(ctx_id=0, det_size=(1600, 1600), det_thresh=0.10)

def best_face(img):
    faces = app.get(img)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

def crop_from_original(path: Path):
    """
    원본에서만 얼굴 crop 생성(보충용)
    - 얼굴 bbox가 너무 작으면 skip (전신샷/배경샷 방지)
    - norm_crop으로 정렬 crop 만들고 OUT_SIZE로 리사이즈
    """
    img = imread_unicode(path)
    if img is None:
        return None, "unreadable"

    face = best_face(img)
    if face is None or getattr(face, "kps", None) is None:
        return None, "no_face"

    # 얼굴이 너무 작으면 버림
    x1, y1, x2, y2 = face.bbox
    face_area = max(0, (x2-x1)) * max(0, (y2-y1))
    img_area  = img.shape[0] * img.shape[1]
    if img_area <= 0:
        return None, "bad_image"
    if (face_area / img_area) < MIN_FACE_RATIO_IN_ORIGINAL:
        return None, "face_too_small"

    crop = norm_crop(img, face.kps)  # 112
    if OUT_SIZE != 112:
        crop = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)

    if is_bad_crop_simple(crop):
        return None, "bad_crop_simple"

    return crop, None

def crop_is_valid(crop_img):
    """
    핵심: 기존 crop은 최대한 살린다.
    - 기본은 단순 품질 체크만
    - 얼굴 재검출은 OFF(필요하면 켤 수 있음)
    """
    if crop_img is None:
        return False
    if is_bad_crop_simple(crop_img):
        return False

    if not USE_FACE_RECHECK_ON_CROP:
        return True

    # (옵션) crop에서 재검출 시도: 업스케일해서 성공률 올림
    up = cv2.resize(crop_img, (512, 512), interpolation=cv2.INTER_CUBIC)
    f = best_face(up)
    if f is None:
        return True  # 재검출 실패만으로 버리지 않음(완화)
    x1,y1,x2,y2 = f.bbox
    area = max(0, (x2-x1)) * max(0, (y2-y1))
    if area / (512*512) < 0.10:
        return False
    return True

# ======================
# 3) 기존 crop 검수 (원본 crop 폴더는 건드리지 않고, bad는 "복사"만)
# ======================
def audit_existing_crops(crop_cls_dir: Path, split_name: str):
    kept_imgs = []
    bad_paths = []

    for p in sorted(list_images(crop_cls_dir), key=lambda x: x.name):
        img = imread_unicode(p)
        if not crop_is_valid(img):
            bad_paths.append(p)
        else:
            kept_imgs.append(img)

    # bad로 "복사/이동" (기본 copy)
    for p in bad_paths:
        dst = BAD_DIR / split_name / crop_cls_dir.name / p.name
        safe_put(p, dst, BAD_ACTION)

    return kept_imgs, len(bad_paths)

# ======================
# 4) 부족분 보충 (원본 풀에서 얼굴 성공한 것만)
# ======================
def refill_from_sources(cls_name: str, need: int, used_crop_hashes: set, used_file_hashes: set, source_dirs: list, split_name: str):
    added = []

    candidates = []
    for d in source_dirs:
        cls_dir = d / cls_name
        if cls_dir.exists():
            candidates += list_images(cls_dir)

    random.shuffle(candidates)

    fail_cls = FAIL_DIR / split_name / cls_name
    fail_cls.mkdir(parents=True, exist_ok=True)

    for p in candidates:
        if need <= 0:
            break

        # 동일 원본 파일 중복 방지
        try:
            fh = file_hash(p)
        except:
            continue
        if fh in used_file_hashes:
            continue

        crop, reason = crop_from_original(p)
        if crop is None:
            # 실패 원본은 참고용으로 저장(copy)
            dst = fail_cls / p.name
            safe_put(p, dst, FAIL_ACTION)
            continue

        ch = img_hash(crop)
        if ch in used_crop_hashes:
            continue

        used_file_hashes.add(fh)
        used_crop_hashes.add(ch)
        added.append(crop)
        need -= 1

    return added

# ======================
# 5) 저장 + 재넘버링
# ======================
def save_numbered(out_dir: Path, cls_name: str, crops: list, target_n: int):
    out_cls = out_dir / cls_name
    out_cls.mkdir(parents=True, exist_ok=True)

    crops = crops[:target_n]
    for i, img in enumerate(crops, start=1):
        imwrite_unicode_jpg(out_cls / f"Image_{i:02d}.jpg", img)

# ======================
# 6) split 처리
# ======================
def fix_split(split_name: str, crop_dir: Path, out_dir: Path, target_n: int, source_dirs: list):
    clear_dir(out_dir)
    BAD_DIR.mkdir(parents=True, exist_ok=True)
    FAIL_DIR.mkdir(parents=True, exist_ok=True)

    crop_classes = [d for d in crop_dir.iterdir() if d.is_dir()]
    if not crop_classes:
        raise RuntimeError(f"No class folders in {crop_dir}")

    print(f"\n=== Fix {split_name} (v2) ===")

    for cls in sorted(crop_classes, key=lambda x: x.name):
        cls_name = cls.name

        # (A) 기존 crop을 최대한 살림
        kept, bad_cnt = audit_existing_crops(cls, split_name)

        # 중복 방지용 해시
        used_crop_hashes = set(img_hash(img) for img in kept)
        used_file_hashes = set()

        # (B) 부족분만 보충
        need = target_n - len(kept)
        if need > 0:
            added = refill_from_sources(
                cls_name=cls_name,
                need=need,
                used_crop_hashes=used_crop_hashes,
                used_file_hashes=used_file_hashes,
                source_dirs=source_dirs,
                split_name=split_name
            )
            kept += added

        # (C) 저장
        if len(kept) < target_n:
            print(f"[WARN] {split_name} {cls_name}: saved={len(kept)} < target={target_n} (원본 더 필요)")
            save_numbered(out_dir, cls_name, kept, len(kept))
        else:
            save_numbered(out_dir, cls_name, kept, target_n)

        print(f"{split_name} {cls_name}: bad_copied={bad_cnt}, final_saved={min(len(kept), target_n)}")

# ======================
# 실행
# ======================
fix_split(
    split_name="train",
    crop_dir=TRAIN_CROP_DIR,
    out_dir=TRAIN_OUT_DIR,
    target_n=TRAIN_TARGET,
    source_dirs=[TRAIN_DIR, TRAIN_COPY_DIR]
)

test_sources = []
if TEST_DIR.exists():
    test_sources.append(TEST_DIR)
test_sources.append(TRAIN_COPY_DIR)

fix_split(
    split_name="test",
    crop_dir=TEST_CROP_DIR,
    out_dir=TEST_OUT_DIR,
    target_n=TEST_TARGET,
    source_dirs=test_sources
)

print("\nDone (v2).")
print("Train out:", TRAIN_OUT_DIR.resolve())
print("Test out :", TEST_OUT_DIR.resolve())
print("Bad saved:", BAD_DIR.resolve())
print("Fail saved:", FAIL_DIR.resolve())