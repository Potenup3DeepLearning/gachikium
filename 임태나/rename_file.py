import os
import re

# 1. '토끼상' 폴더 경로
folder_path = r'C:\gachikium\dataset\train\도롱뇽상' 

# 2. 파일 목록 가져오기
files = os.listdir(folder_path)

# 3. 현재 이미 있는 'Image_숫자' 중 가장 큰 번호 찾기
max_num = 0
for f in files:
    # 'Image_' 다음에 숫자가 오는 패턴을 찾습니다
    match = re.match(r'Image_(\d+)', f)
    if match:
        num = int(match.group(1))
        if num > max_num:
            max_num = num

print(f" 현재 마지막 번호는 {max_num}번입니다. 이 다음 번호부터 이름을 바꿀게요!")

# 4. 'Image_'로 시작하지 않는 파일들(스크린샷 등)만 골라내기
new_files = [f for f in files if not f.startswith("Image_")]
new_files.sort() # 스크린샷 날짜순/이름순 정렬

# 5. 새 번호 부여 시작
count = 0
for filename in new_files:
    max_num += 1 # 기존 마지막 번호 다음부터 시작
    ext = os.path.splitext(filename)[1]
    new_name = f"Image_{max_num}{ext}"
    
    src = os.path.join(folder_path, filename)
    dst = os.path.join(folder_path, new_name)
    
    os.rename(src, dst)
    count += 1

print(f"✅ 새 파일 {count}개의 정리가 완료되었습니다! (최종 번호: {max_num})")