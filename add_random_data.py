import numpy as np
import pandas as pd
import add_data_lib
import s_e_make
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pkl_path", "-pkl", type=str, default="feature_v5_fc7.pkl")
parser.add_argument("--data_num", "-n", type=int, default=50)
parser.add_argument('--out_np_file_path', '-np', type=str, default='./npy_files/')
parser.add_argument('--remove_np_file_path', '-re', type=str, default=None)
args = parser.parse_args()

pkl_path = args.pkl_path
data_num = args.data_num
out_np_file_path = args.out_np_file_path
remove_np_file_path = args.remove_np_file_path

old_data = []
#特徴量のデータフレームを読み込み
df = pd.read_pickle(pkl_path)
#データフレームをクラス毎に分割
if remove_np_file_path is not None:
    old_data = np.load(remove_np_file_path)
    #データフレームから既に存在する画像データのリストを削除
    df = add_data_lib.remove_existing_data(df, remove_np_file_path)

df_list = add_data_lib.df_by_class(df)
#データフレームに算出した特徴量の数を追加
add_num_df = []
for df_i in df_list:
    df_i = add_data_lib.reset_index(df_i)
    df_i = add_data_lib.add_fe_num(df_i)
    add_num_df.append(df_i)

#ランダムに次に追加するデータを選択
add_data_name = add_data_lib.add_random_data(add_num_df, data_num)
now_data = list(old_data) + list(add_data_name)
print("データ数： {}".format(len(now_data)))

npy_file_path = out_np_file_path
np.save(npy_file_path, now_data)
print(npy_file_path + "　に保存しました")
