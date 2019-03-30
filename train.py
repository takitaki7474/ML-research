# -*- coding:utf-8 -*-

import numpy as np
import cv2
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import iterators, training, optimizers, serializers
from chainer.datasets import tuple_dataset, split_dataset_random
from chainer.training import extensions
import argparse
import os
import cnn_mynet
import make_train_data

#コマンドライン引数をパースする
parser = argparse.ArgumentParser(description='Train Sample')
parser.add_argument('--train_list', '-train', default='./data_v5', type=str, help='Train image folder name')
parser.add_argument('--model_name', '-m', default='v1.model', type=str, help='model name')
parser.add_argument('--epoch', '-e', type=int, default=300, help='Number of epochs to train')
parser.add_argument('--batchsize', '-b', type=int, default=128, help='Number of batchsize to train')
parser.add_argument('--alpha', '-a', type=float, default=0.001, help='Number of alpha to train')
parser.add_argument('--numpy_file', '-np', type=str, default='random.npy', help='Number of data to train')
parser.add_argument('--pkl_file', '-pkl', type=str, default='feature_v5_fc7.pkl', help='Number of data to train')
parser.add_argument('--gpu_id', '-gpu', type=int, default=-1)
args = parser.parse_args()

img_folder = args.train_list
max_epoch = args.epoch
batchsize = args.batchsize
alpha = args.alpha
model_name = args.model_name
numpy_file = args.numpy_file
pkl_file = args.pkl_file
gpu_id = args.gpu_id

np_file_path = os.path.join("./npy_files", numpy_file)
pkl_file_path = os.path.join("./feature", pkl_file)
save_model_path = os.path.join("./learned_model", model_name)
save_model_path = save_model_path + ".model"
test_folder_path = "./test_img_v2"

train_list, train_image_list = make_train_data.make_train_list(img_folder, pkl_file_path, np_file_path)
val_list = make_train_data.make_test_data(test_folder_path)
#val_list, val_image_list = make_train_data.make_val_list(img_folder, pkl_file_path, np_file_path)


x_train, y_train = make_train_data.make_dataset(train_list)
x_val, y_val = make_train_data.make_dataset(val_list)
#データセットを構築
train_data = tuple_dataset.TupleDataset(x_train, y_train)
val_data = tuple_dataset.TupleDataset(x_val, y_val)

train_iter = iterators.SerialIterator(train_data, batchsize)
valid_iter = iterators.SerialIterator(val_data, batchsize, repeat=False, shuffle=False)


net = cnn_mynet.MyNet_6(3)
# ネットワークをClassifierで包んで、ロスの計算などをモデルに含める
net = L.Classifier(net)

# 最適化手法の選択
optimizer = optimizers.Adam(alpha=alpha).setup(net)

# UpdaterにIteratorとOptimizerを渡す
updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

# TrainerにUpdaterを渡す
trainer = training.Trainer(updater, (max_epoch, 'epoch'), out="./result/" + model_name) #変更点

trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.Evaluator(valid_iter, net, device=gpu_id), name="val")
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'l1/W/data/std', 'elapsed_time']))
trainer.extend(extensions.PlotReport(['l1/W/data/std'], x_key='epoch', file_name='std.png'))
trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
#trainer.extend(extensions.PlotReport(['val/main/loss'], x_key='epoch', file_name='loss.png'))
#trainer.extend(extensions.PlotReport(['val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.extend(extensions.dump_graph('main/loss'))

trainer.run()

serializers.save_npz(save_model_path, net)  #変更点
