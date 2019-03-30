import pandas as pd
import math
import numpy as np
import os
import glob

def loss_make(log_file_path):
    main_loss = {}
    val_main_loss = {}
    for log_file in log_file_path:
        log_file_json = pd.read_json(log_file)
        main_loss[log_file] = log_file_json["main/loss"].values
        val_main_loss[log_file] = log_file_json["val/main/loss"].values

    return main_loss, val_main_loss

def s_make(main_loss, val_main_loss, log_file_path):
    x = np.arange(0,300)
    y = 0.5 ** x
    main_loss_s = {}
    val_main_loss_s = {}

    for log_file in log_file_path:
        main_loss_store = main_loss[log_file]
        val_main_loss_store = val_main_loss[log_file]
        diff_main_loss = main_loss_store - y
        diff_val_main_loss = val_main_loss_store - y
        main_loss_s[log_file] = np.sum(diff_main_loss[diff_main_loss>0])
        val_main_loss_s[log_file] = np.sum(diff_val_main_loss[diff_val_main_loss>0])

    return main_loss_s, val_main_loss_s

def e1_cul(loss_list):
    diff_loss = []
    for i in range(len(loss_list)):
        if i+1 <= len(loss_list)-1:
            diff_loss.append(loss_list[i+1] - loss_list[i])

    diff_loss = np.asarray(diff_loss)

    e = np.sum(diff_loss[diff_loss>0])
    return e

def e1_make(main_loss, val_main_loss, log_file_path):
    main_loss_e = {}
    val_main_loss_e = {}

    for log_file in log_file_path:
        main_loss_store = main_loss[log_file]
        val_main_loss_store = val_main_loss[log_file]
        main_loss_e[log_file] = e1_cul(main_loss_store)
        val_main_loss_e[log_file] = e1_cul(val_main_loss_store)

    return main_loss_e, val_main_loss_e

def e2_cul(loss_list):
    diff_loss = []
    for i in range(len(loss_list)):
        if i+1 <= len(loss_list)-1:
            diff_loss.append(loss_list[i+1] - loss_list[i])

    diff_loss = np.asarray(diff_loss)

    e = np.sum(diff_loss>0)
    return e

def e2_make(main_loss, val_main_loss, log_file_path):
    main_loss_e = {}
    val_main_loss_e = {}

    for log_file in log_file_path:
        main_loss_store = main_loss[log_file]
        val_main_loss_store = val_main_loss[log_file]
        main_loss_e[log_file] = e2_cul(main_loss_store)
        val_main_loss_e[log_file] = e2_cul(val_main_loss_store)

    return main_loss_e, val_main_loss_e

def make_s_e_list(main_loss, val_main_loss, log_file_path):
    train_s = []
    train_e = []
    val_s = []
    val_e = []

    train_s_dict, val_s_dict = s_make(main_loss, val_main_loss, log_file_path)
    train_e_dict, val_e_dict = e1_make(main_loss, val_main_loss, log_file_path)
    train_s = list(train_s_dict.values())
    train_e = list(train_e_dict.values())
    val_s = list(val_s_dict.values())
    val_e = list(val_e_dict.values())

    return (train_s, train_e, val_s, val_e)
