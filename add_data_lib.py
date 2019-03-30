import pandas as pd
import numpy as np

def remove_existing_data(df, np_file_path):
    img_list = np.load(np_file_path)
    remove_index = []

    for img_name in img_list:
        remove_index.append(df[df.img == img_name].index[0])

    removed_df = df.drop(index=remove_index)
    return removed_df

def df_by_class(df):
    df_list = []
    class_num = len(np.unique(df.label.values))
    for class_i in range(class_num):
        df_class_i = df[df.label == int(class_i)]
        df_list.append(df_class_i)

    return df_list

def reset_index(df):
    df_reset = df.reset_index(drop=True)
    return df_reset

def add_fe_num(df):
    fe_num_list = []

    for df_index in range(len(df)):
        fe_num_list.append(np.sum(df.feature.values[df_index] > 0))

    df_fe_num = df.copy()
    df_fe_num["fe_num"] = fe_num_list
    df_fe_num = df_fe_num.sort_values("fe_num")
    df_fe_num = reset_index(df_fe_num)

    return df_fe_num

def add_data_select(df_list, feature_stage, add_img_num):
    fe_stage_table = {1:[600,700], 2:[700,800], 3:[800,900]}
    fe_range = fe_stage_table[feature_stage]

    add_data_name = []

    for label in range(len(df_list)):
        df = df_list[label]
        add_data_name.extend(df.img[(df.fe_num>fe_range[0])&(df.fe_num<=fe_range[1])].sample(n=add_img_num).values)

    return add_data_name

def add_random_data(df_list, add_img_num):
    add_data_name = []

    for label in range(len(df_list)):
        df = df_list[label]
        add_data_name.extend(df.img.sample(n=add_img_num).values)

    return add_data_name
