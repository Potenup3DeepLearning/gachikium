import os, shutil, random, hashlib
from pathlib import Path
from collections import defaultdict

# =========================
# 설정
# =========================
DATASET_DIR = Path(r"C:\gachikium\dataset\train_copy")  # ✅ 사본 폴더로 바꿔!
TARGET_PER_CLASS = 80
SEED = 42

# 삭제 대신 이동(복구 가능)
REMOVED_DIR = DATASET_DIR.parent / (DATASET_DIR.name + "_removed")
MOVE_INSTEAD_OF_DELETE = True

# 이미지 확장자
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

random.seed(SEED)

# =========================
# 유틸
# =========================
def file_hash(path: Path, chunk_size=1024 * 1024) -> str:
    """이미지 '내용' 기반 중복 탐지용 해시(빠르고 안정적)"""
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def safe_remove(path: Path, reason: str):
    """삭제 대신 removed 폴더로 이동"""
    if MOVE_INSTEAD_OF_DELETE:
        rel = path.relative_to(DATASET_DIR)
        dst = REMOVED_DIR / reason / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        # 혹시 같은 이름 충돌 시 _1, _2 붙이기
        if dst.exists():
            stem, suf = dst.stem, dst.suffix
            i = 1
            while (dst.parent / f"{stem}_{i}{suf}").exists():
                i += 1
            dst = dst.parent / f"{stem}_{i}{suf}"

        shutil.move(str(path), str(dst))
    else:
        path.unlink()

def list_images(cls_dir: Path):
    return [p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

# =========================
# 메인
# =========================
if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Not found: {DATASET_DIR}")

REMOVED_DIR.mkdir(parents=True, exist_ok=True)

class_dirs = [d for d in DATASET_DIR.iterdir() if d.is_dir()]

report = []
shortages = []

for cls_dir in sorted(class_dirs, key=lambda p: p.name):
    imgs = list_images(cls_dir)

    if not imgs:
        report.append(f"[WARN] {cls_dir.name}: 이미지 없음")
        shortages.append((cls_dir.name, TARGET_PER_CLASS))
        continue

    # 1) 파일명 중복 체크(동일 폴더 내)
    name_map = defaultdict(list)
    for p in imgs:
        name_map[p.name].append(p)

    # 같은 이름이 2개 이상이면, 1개만 남기고 나머지 제거(removed로)
    removed_name_dups = 0
    for name, paths in name_map.items():
        if len(paths) > 1:
            # 하나는 남기고 나머지 제거
            # (어차피 같은 이름이면 덮어쓰기 못하고 공존했을 가능성 낮지만, 안전장치)
            for extra in paths[1:]:
                safe_remove(extra, reason="dup_name")
                removed_name_dups += 1

    # 갱신
    imgs = list_images(cls_dir)

    # 2) 내용(해시) 기반 중복 체크 (같은 이미지가 여러 파일명으로 있을 수 있음)
    hash_map = {}
    removed_hash_dups = 0

    for p in imgs:
        try:
            h = file_hash(p)
        except Exception:
            # 읽기 실패 파일은 제거(깨진 파일 등)
            safe_remove(p, reason="corrupt_or_unreadable")
            continue

        if h in hash_map:
            # 이미 같은 내용의 이미지가 있음 → 중복 제거
            safe_remove(p, reason="dup_content")
            removed_hash_dups += 1
        else:
            hash_map[h] = p

    # 갱신
    imgs = list_images(cls_dir)

    # 3) 목표 개수로 맞추기
    n = len(imgs)
    removed_excess = 0

    if n > TARGET_PER_CLASS:
        random.shuffle(imgs)
        excess = imgs[TARGET_PER_CLASS:]
        for p in excess:
            safe_remove(p, reason="excess_trim")
            removed_excess += 1

    # 최종 개수
    final_n = len(list_images(cls_dir))

    # 4) 부족 리포트 (부족분을 “삭제/추가”할 수는 없으니 알려만 줌)
    if final_n < TARGET_PER_CLASS:
        shortages.append((cls_dir.name, TARGET_PER_CLASS - final_n))

    report.append(
        f"{cls_dir.name}: start={n} "
        f"(rm_name_dup={removed_name_dups}, rm_content_dup={removed_hash_dups}, rm_excess={removed_excess}) "
        f"-> final={final_n}"
    )

print("\n".join(report))

if shortages:
    print("\n[부족 클래스]")
    for name, lack in shortages:
        print(f"- {name}: {lack}장 부족")
else:
    print("\n모든 클래스가 목표 수량을 충족합니다.")

print("\nRemoved moved to:", REMOVED_DIR)