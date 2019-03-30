import pandas as pd
import math
import numpy as np
import os
import glob
from scipy import optimize
from scipy.integrate import quad

def fit_func3(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d
def fit_func6(x,a,b,c,d,e,f,g):
    return a*x**6 + b*x**5 + c*x**4 + d*x**3 + e*x**2 + f*x + g
def y1(x):
    return 0.5 ** x

def loss_make(log_file_path):
    main_loss = {}
    val_main_loss = {}
    for log_file in log_file_path:
        log_file_json = pd.read_json(log_file)
        main_loss[log_file] = log_file_json["main/loss"].values
        val_main_loss[log_file] = log_file_json["val/main/loss"].values

    return main_loss, val_main_loss

def s_cal(func, px, y, loss_dic):
    res = optimize.curve_fit(func, px, loss_dic)
    k = []
    n = 6
    for i in range(n + 1):
        k.append(res[0][i])
    px2 = []
    for x in px:
        px2.append(fit_func6(x,k[0],k[1],k[2],k[3],k[4],k[5],k[6]))
    px2 = np.array(px2)
    def diff_px2(x):
        return fit_func6(x,k[0],k[1],k[2],k[3],k[4],k[5],k[6]) - y1(x)
    I, _ = quad(diff_px2, 0, 300)

    return I

def s_make(main_loss, val_main_loss, log_file_path):
    px = np.arange(0,300)
    y = y1(px)
    main_loss_s = {}
    val_main_loss_s = {}

    for log_file in log_file_path:
        main_loss_s[log_file] = s_cal(fit_func6, px, y, main_loss[log_file])
        val_main_loss_s[log_file] = s_cal(fit_func6, px, y, val_main_loss[log_file])

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
