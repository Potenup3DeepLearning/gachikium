import cv2
import os

img = cv2.imread(r"C:\gachikium\images/puppy.png")

print(img.shape)

rows = 5
cols = 8

h, w, _ = img.shape

cell_h = h // rows
cell_w = w // cols

os.makedirs("faces", exist_ok=True)

count = 0

for r in range(rows):
    for c in range(cols):

        y1 = r * cell_h
        y2 = (r + 1) * cell_h

        x1 = c * cell_w
        x2 = (c + 1) * cell_w

        crop = img[y1:y2, x1:x2]

        cv2.imwrite(f"faces/face_{count}.jpg", crop)

        count += 1

print("저장된 이미지 수:", count)