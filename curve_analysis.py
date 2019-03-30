# -*- coding:utf-8 -*-

import s_e_make
import argparse
import logging
# from pythonjsonlogger import jsonlogger

parser = argparse.ArgumentParser()
parser.add_argument("--in_log_path", "-i", type=str, default="./result/")
parser.add_argument("--out_log_path", "-o", type=str, default="./analysis_log/")
parser.add_argument("--data_num", "-n", type=int, default=10000000)
args = parser.parse_args()

in_log_path = args.in_log_path
out_log_path = args.out_log_path
data_num = args.data_num

log_file_path = [in_log_path]
main_loss, val_main_loss = s_e_make.loss_make(log_file_path)
train_s, train_e, val_s, val_e = s_e_make.make_s_e_list(main_loss, val_main_loss, log_file_path)


# ログの出力名を設定
logger = logging.getLogger('LoggingTest')

# ログレベルの設定
logger.setLevel(10)

# ログのファイル出力先を設定
fh = logging.FileHandler(out_log_path)
logger.addHandler(fh)

# ログのコンソール出力の設定
sh = logging.StreamHandler()
logger.addHandler(sh)

# ログの出力形式の設定
formatter = logging.Formatter('%(asctime)s:%(message)s')
fh.setFormatter(formatter)
sh.setFormatter(formatter)

logger.log(20, "データ数：" + str(data_num))
logger.log(20, "訓練誤差の面積: " + str(train_s))
logger.log(20, "訓練誤差の振れ: " + str(train_e))
logger.log(20, "汎化誤差の面積: " + str(val_s))
logger.log(20, "汎化誤差の振れ: " + str(val_e))
logger.log(20, "")

'''
logger = logging.getLogger("loggingTest")
logger.setLevel(10)
h = logging.FileHandler(out_log_path)
logger.addHandler(h)
sh = logging.StreamHandler()
logger.addHandler(sh)
h.setFormatter(jsonlogger.JsonFormatter())

logger.info(log_dic)
'''
