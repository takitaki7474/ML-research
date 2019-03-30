# -*- coding:utf-8 -*-

import cv2
import datetime
import argparse
import time
import os

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--folder', '-f', default='./new_img/10', type=str,help="output folder name, example ./new_img/10")
parser.add_argument('--img_num', '-n', default=100, type=int)
args = parser.parse_args()


cap = cv2.VideoCapture(0)

count = 0
count_imwrite = 0
frame_span = 10
images = args.img_num#撮影枚数

while True:
    ret, frame = cap.read()
    now = datetime.datetime.now()

    if count % frame_span == 0:
        img_path = os.path.join(args.folder, str(now) + '.png')
        cv2.imwrite(img_path,frame)
        count_imwrite += 1

    count += 1

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) == 27:
        break
    if count == (images*frame_span):
        break

cap.release()

print(time.time()-start)
