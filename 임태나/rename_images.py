import os
from pathlib import Path

# 이미지 폴더 경로
folder = Path(r"C:\gachikium\dataset\train\쿼카상")

# 이미지 확장자
exts = (".jpg", ".jpeg", ".png", ".webp")

# 이미지 목록
images = [p for p in folder.iterdir() if p.suffix.lower() in exts]

# 정렬
images = sorted(images)

# 1단계: 임시 이름으로 변경 (충돌 방지)
temp_paths = []
for i, img in enumerate(images):
    temp_name = folder / f"temp_{i:03d}{img.suffix}"
    img.rename(temp_name)
    temp_paths.append(temp_name)

# 2단계: 최종 이름으로 변경
for i, img in enumerate(sorted(temp_paths)):
    new_name = folder / f"Image_{i+1:02d}{img.suffix}"
    img.rename(new_name)

print("파일 이름 정렬 완료")
print(f"총 이미지 수: {len(images)}")