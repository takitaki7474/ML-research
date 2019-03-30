import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_folder","-im",default="./cut_img_3",type=str)
args = parser.parse_args()

img_folder = args.img_folder

# ルックアップテーブルの生成
min_table = 50
max_table = 205
diff_table = max_table - min_table

LUT_HC = np.arange(256, dtype = 'uint8' )
LUT_LC = np.arange(256, dtype = 'uint8' )

# ハイコントラストLUT作成
for i in range(0, min_table):
    LUT_HC[i] = 0
for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table
for i in range(max_table, 255):
    LUT_HC[i] = 255
# ローコントラストLUT作成
for i in range(256):
    LUT_LC[i] = min_table + i * (diff_table) / 255

img_list = os.listdir(img_folder)
if ".DS_Store" in img_list:
    img_list.remove(".DS_Store")
img_list.sort()

for i,img in enumerate(img_list):
    img_path = os.path.join(img_folder,img)
    src = cv2.imread(img_path)
    if i%2 == 0:
        high_cont_img = cv2.LUT(src, LUT_HC)
        write_path = os.path.join(img_folder, "n_" + str(i) + ".png")
        cv2.imwrite(write_path, high_cont_img)
    elif i%2 == 1:
        low_cont_img = cv2.LUT(src, LUT_LC)
        write_path = os.path.join(img_folder, "n_" + str(i) + ".png")
        cv2.imwrite(write_path, low_cont_img)
