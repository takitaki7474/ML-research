import pandas as pd
import numpy as np
import os

#ランダムなtrainデータを生成
#入力：１クラスのデータ数data_num
def random_select(df, out_npy_name, data_num, class_num):
    random_data = []

    for class_i in range(class_num):
        df_class_i = df[df.label == int(class_i)]
        img_list = df_class_i.sample(n=data_num).img.values
        random_data.extend(img_list)

    npy_file_path = os.path.join("./npy_files", out_npy_name)

    np.save(npy_file_path, random_data)

def remove_df(df, in_npy):
    removed_index = []
    np_file_path = os.path.join("./npy_files", in_npy)
    img_list = np.load(np_file_path)
    for img in img_list:
        removed_index.append(df[df.img == img].index[0])
    removed_df = df.drop(index = removed_index)
    return removed_df

#入力：１クラスごとのdataframe、 クエリの数
#出力：画像名のリスト
def random_center(df_i, center_num):

    query_list = df_i.sample(n=center_num).img.values

    return query_list

#配列同士の距離を算出
def euclid(center, data):
    diff = center-data
    sq = np.square(diff)
    total = np.sum(sq)
    distance = np.sqrt(total)
    return distance

# 入力： distを含まないdataframe, クラス数class_num
# 出力： クラスごとのdataframe
def df_by_class(df, class_num):
    df_list = []
    for class_i in range(class_num):
        df_class_i = df[df.label == int(class_i)]
        df_list.append(df_class_i)

    return df_list

#入力：distを含まないdataframe、クエリー画像名query_img
#出力：distを追加したdataframe
def df_add_distance(df, query_img):
    distance = []
    count = 0

    for index in range(len(df.feature)):
        query_img_feature = df[df.img == query_img].feature.values[0]
        d = euclid(query_img_feature, df.feature.values[index])
        distance.append(np.array(d,dtype="float32"))

    df_distance = df.copy()
    df_distance["dist"] = distance
    added_distance_df = df_distance.sort_values("dist")

    return added_distance_df

#入力：distを含むdataframe, データ数num
#出力: 画像名のリストimg_list, 半径radius
def select_data(df, num):
    img_list = df.img[1:num+1].values
    radius = df.dist.values[num]

    return img_list, radius

#訓練データにimg_listを加える
def add_data(img_list, in_npy_name, out_npy_name):
    save_data = []
    in_path = os.path.join("./npy_files", in_npy_name)
    out_path = os.path.join("./npy_files",out_npy_name)
    train_data = np.load(in_path)
    save_data.extend(train_data)
    save_data.extend(img_list)
    np.save(out_path, save_data)
