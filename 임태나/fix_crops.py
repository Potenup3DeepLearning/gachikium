import cv2, numpy as np, random, hashlib, shutil
from pathlib import Path
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

# ======================
# 0) 너 폴더에 맞게 수정
# ======================
TRAIN_COPY_DIR = Path(r"C:\gachikium\dataset\train_copy")   # 클래스당 80
TRAIN_DIR      = Path(r"C:\gachikium\dataset\train")        # 클래스당 70
# ※ 원본 test 폴더가 있으면 넣어줘. 없으면 아래에서 train_copy에서 보충하는 방식으로 동작(테스트셋이 바뀔 수 있음)
TEST_DIR       = Path(r"C:\gachikium\dataset\test")         # (있으면 좋음) 클래스당 14

TRAIN_CROP_DIR = Path(r"C:\gachikium\dataset\train_crop")   # 현재 크롭 결과(56)
TEST_CROP_DIR  = Path(r"C:\gachikium\dataset\test_crop")    # 현재 크롭 결과(14)

TRAIN_OUT_DIR  = Path(r"C:\gachikium\dataset\train_crop_fixed")
TEST_OUT_DIR   = Path(r"C:\gachikium\dataset\test_crop_fixed")

BAD_DIR        = Path(r"C:\gachikium\dataset\crop_bad")     # 불량 크롭 이동
FAIL_DIR       = Path(r"C:\gachikium\dataset\crop_fail")    # 보충 시도 실패 원본 복사

TRAIN_TARGET = 56
TEST_TARGET  = 14
OUT_SIZE = 224
SEED = 42

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
            if not b: break
            h.update(b)
    return h.hexdigest()

def is_bad_crop(img, dark_thresh=12, var_thresh=25):
    # 검은 막대/너무 어두움/너무 단조(블러/배경) 걸러내기
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.mean() < dark_thresh:
        return True
    if gray.var() < var_thresh:
        return True
    return False

# ======================
# 2) 얼굴 검출기
# ======================
app = FaceAnalysis(name="buffalo_l")
# GPU 없으면 ctx_id=-1로 바꿔도 됨
app.prepare(ctx_id=0, det_size=(1280, 1280), det_thresh=0.15)

def best_face(img):
    faces = app.get(img)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

def crop_from_original(path: Path):
    img = imread_unicode(path)
    if img is None:
        return None, "unreadable"

    face = best_face(img)
    if face is None or getattr(face, "kps", None) is None:
        return None, "no_face"

    crop = norm_crop(img, face.kps)  # 112
    if OUT_SIZE != 112:
        crop = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)

    if is_bad_crop(crop):
        return None, "bad_crop"
    return crop, None

def crop_is_valid(crop_img):
    # 크롭 이미지 자체에서 다시 얼굴이 잡히는지 확인(강력한 품질 체크)
    f = best_face(crop_img)
    if f is None:
        return False
    # 너무 작은 얼굴(대부분 배경) 방지
    x1, y1, x2, y2 = f.bbox
    area = max(0, (x2-x1)) * max(0, (y2-y1))
    if area < (OUT_SIZE * OUT_SIZE) * 0.10:  # 얼굴 bbox가 이미지의 10% 미만이면 불량 취급
        return False
    if is_bad_crop(crop_img):
        return False
    return True

# ======================
# 3) (A) 기존 crop 검수 -> 합격만 모으기
# ======================
def audit_existing_crops(crop_cls_dir: Path, split_name: str):
    ok = []   # (crop_img, src_path)
    bad = []  # src_path

    for p in sorted(list_images(crop_cls_dir), key=lambda x: x.name):
        img = imread_unicode(p)
        if img is None or not crop_is_valid(img):
            bad.append(p)
        else:
            ok.append((img, p))
    # 불량 이동(원본 보존을 위해 crop만 이동)
    for p in bad:
        dst = BAD_DIR / split_name / crop_cls_dir.name / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dst))
    return ok, len(bad)

# ======================
# 4) (B) 부족분 보충: 원본 풀에서 얼굴 성공한 것만 채우기
# ======================
def refill_from_sources(cls_name: str, need: int, used_crop_hashes: set, used_file_hashes: set, source_dirs: list, split_name: str):
    added = []

    # 후보 파일 목록 구성
    candidates = []
    for d in source_dirs:
        cls_dir = d / cls_name
        if not cls_dir.exists():
            continue
        candidates += list_images(cls_dir)

    random.shuffle(candidates)

    fail_cls = FAIL_DIR / split_name / cls_name
    fail_cls.mkdir(parents=True, exist_ok=True)

    for p in candidates:
        if need <= 0:
            break

        # 동일 파일(내용) 중복 방지
        try:
            fh = file_hash(p)
        except:
            continue
        if fh in used_file_hashes:
            continue

        crop, reason = crop_from_original(p)
        if crop is None:
            # 실패 원본은 복사해두면 나중에 왜 실패했는지 볼 수 있음(선택)
            try:
                shutil.copy2(p, fail_cls / p.name)
            except:
                pass
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
# 5) (C) 최종 저장 + 재넘버링
# ======================
def save_numbered(out_dir: Path, cls_name: str, crops: list, target_n: int):
    out_cls = out_dir / cls_name
    out_cls.mkdir(parents=True, exist_ok=True)

    # target_n만 저장
    crops = crops[:target_n]
    for i, img in enumerate(crops, start=1):
        imwrite_unicode_jpg(out_cls / f"Image_{i:02d}.jpg", img)

# ======================
# 6) 메인: split별로 클래스 반복
# ======================
def fix_split(split_name: str, crop_dir: Path, out_dir: Path, target_n: int, source_dirs: list):
    clear_dir(out_dir)

    crop_classes = [d for d in crop_dir.iterdir() if d.is_dir()]
    if not crop_classes:
        raise RuntimeError(f"No class folders in {crop_dir}")

    print(f"\n=== Fix {split_name} ===")

    for cls in sorted(crop_classes, key=lambda x: x.name):
        cls_name = cls.name

        # 1) 기존 크롭 검수
        ok_list, bad_cnt = audit_existing_crops(cls, split_name)

        # ok_list -> crop_img들만 뽑기
        kept = [img for (img, _) in ok_list]

        # 해시 셋 구성(중복 방지)
        used_crop_hashes = set(img_hash(img) for img in kept)
        used_file_hashes = set()  # 원본 파일 해시 중복 방지(원본 풀에서 가져올 때 사용)

        # 2) 부족분 보충
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

        # 3) 그래도 부족하면 경고
        if len(kept) < target_n:
            print(f"[STOP] {split_name} {cls_name}: kept={len(kept)} < target={target_n}")
            print("→ 이 클래스는 원본 풀(얼굴 잘 나온 사진)을 더 추가해야 함.")
            # 그래도 가능한 만큼은 저장
            save_numbered(out_dir, cls_name, kept, len(kept))
        else:
            save_numbered(out_dir, cls_name, kept, target_n)

        print(f"{split_name} {cls_name}: bad_removed={bad_cnt}, final_saved={min(len(kept), target_n)}")

# ======================
# 실행
# ======================
BAD_DIR.mkdir(parents=True, exist_ok=True)
FAIL_DIR.mkdir(parents=True, exist_ok=True)

# train 보충 소스: train(70) -> train_copy(80)
fix_split(
    split_name="train",
    crop_dir=TRAIN_CROP_DIR,
    out_dir=TRAIN_OUT_DIR,
    target_n=TRAIN_TARGET,
    source_dirs=[TRAIN_DIR, TRAIN_COPY_DIR]
)

# test 보충 소스: test(있으면) -> train_copy(80) (최후 수단)
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

print("\nDone.")
print("Train out:", TRAIN_OUT_DIR.resolve())
print("Test out :", TEST_OUT_DIR.resolve())
print("Bad moved:", BAD_DIR.resolve())
print("Fail copy:", FAIL_DIR.resolve())