import argparse
import numpy as np
import pandas as pd
import cv2
import os
import time
import re
from chainer.links import VGG16Layers


parser = argparse.ArgumentParser(description='Train Sample')
parser.add_argument('--image_folder', '-im', default='./cut_img', type=str, help='input folder')
parser.add_argument('--pkl_file', '-p', default='feature_sub.pkl', type=str, help='pickle file name to write')
args = parser.parse_args()

images_folder = args.image_folder #抽出元の画像フォルダ
file_pkl = args.pkl_file #出力先のpickleファイル名

model = VGG16Layers()

pattern = "^(.D)|^(._.D)"

gpu_id = 0

def extract_feature(class_num, layer_name):

    bottle_label = []
    bottle_img = []
    bottle_feature = []

    for i in range(class_num):
        img_folder_path = os.path.join(images_folder, str(i))
        labels = os.listdir(img_folder_path)
        for img_name in labels:
            test = re.match(pattern, img_name)
            if test != None:
                 labels.remove(img_name)

        labels.sort()
        print(i)
        for img in labels:
            try:
                im_path = os.path.join(img_folder_path, img)
                im = cv2.imread(im_path,1)
                feature = model.extract([im], layers=[layer_name])[layer_name]
                bottle_label.append(np.int32(i))
                bottle_img.append(img)
                bottle_feature.append(np.array(feature.data[0], dtype="float32"))
            except:
                print(im_path)
                print("cannot load image")
                break

    bottle_table = {}
    bottle_table["img"] = bottle_img
    bottle_table["feature"] = bottle_feature
    bottle_table["label"] = bottle_label
    my_df = pd.DataFrame.from_dict(bottle_table)
    file_pkl_path = os.path.join("./feature", file_pkl)
    my_df.to_pickle(file_pkl_path)


if __name__ == "__main__":
    start = time.time()
    extract_feature(3, "fc7")
    print("抽出経過時間 {}秒".format(time.time()-start))
