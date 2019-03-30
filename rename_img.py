import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_folder','-im',default='./images/0/',type=str,help='input image folder name, example ./images/0/')
args = parser.parse_args()

img_folder = args.img_folder

count = 0

files = os.listdir(img_folder)
files.sort()
for file in files:
    new_img_path = os.path.join(img_folder, 'data_2_' + str(count) + '.png')
    old_img_path = os.path.join(img_folder, file)
    os.rename(old_img_path, new_img_path)
    count += 1

print(count)
