#!/usr/bin/env python
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import numpy as np
import queue
import pprint
import random
import argparse
import importlib
import threading
import traceback

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import stdout_to_tqdm
from config import system_configs
from nnet.FuseRoad_factory import NetworkFactory
from torch.multiprocessing import Process, Queue, Pool
from db.datasets import datasets
from torchsummary import summary
from torch.cuda.amp import autocast as autocast, GradScaler

import models.py_utils.misc as utils

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Train CornerNet")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--model_name", help="model name", default="LSTR", type=str)
    parser.add_argument("--iter", dest="start_iter",
                        help="train at iteration i",
                        default=0, type=int)
    parser.add_argument("--threads", dest="threads", default=4, type=int)
    parser.add_argument("--freeze", action="store_true")

    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def prefetch_data(db, queue, sample_data):
    ind = 0
    print("start prefetching data...")
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(db, ind)
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e

def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()

        data["xs"] = [x.pin_memory() for x in data["xs"]]
        data["ys"] = [y.pin_memory() for y in data["ys"]]

        pinned_data_queue.put(data)

        if sema.acquire(blocking=False):
            return

def init_parallel_jobs(dbs, queue, fn):
    tasks = [Process(target=prefetch_data, args=(db, queue, fn)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks

def train(training_dbs, validation_db, training_seg_dbs, validation_seg_db, start_iter=0, freeze=False):
    lane_lr    = system_configs.learning_rate
    seg_lr = system_configs.seg_learning_rate
    max_iteration    = system_configs.max_iter
    pretrained_model = system_configs.pretrain
    snapshot         = system_configs.snapshot
    val_iter         = system_configs.val_iter
    display          = system_configs.display
    decay_rate       = system_configs.decay_rate
    stepsize         = system_configs.stepsize
    batch_size       = system_configs.batch_size

    # getting the size of each database
    training_size   = len(training_dbs[0].db_inds)
    validation_size = len(validation_db.db_inds)

    # queues storing data for training
    training_queue   = Queue(system_configs.prefetch_size) # 5
    training_seg_queue = Queue(system_configs.prefetch_size)
    validation_queue = Queue(5)
    validation_seg_queue = Queue(5)

    # queues storing pinned data for training
    pinned_training_queue   = queue.Queue(system_configs.prefetch_size) # 5
    pinned_training_seg_queue = queue.Queue(system_configs.prefetch_size)
    pinned_validation_queue = queue.Queue(5)
    pinned_validation_seg_queue = queue.Queue(5)

    # load data sampling function
    data_file   = "sample.{}".format(training_dbs[0].data)
    sample_data = importlib.import_module(data_file).sample_data

    seg_data_file = "sample.{}".format(training_seg_dbs[0].data)
    seg_sample_data = importlib.import_module(seg_data_file).sample_data

    # allocating resources for parallel reading
    training_tasks   = init_parallel_jobs(training_dbs, training_queue, sample_data)
    training_seg_tasks = init_parallel_jobs(training_seg_dbs, training_seg_queue, seg_sample_data)
    if val_iter:
        validation_tasks = init_parallel_jobs([validation_db], validation_queue, sample_data)
        validation_seg_tasks = init_parallel_jobs([validation_seg_db], validation_seg_queue, seg_sample_data)

    training_pin_semaphore   = threading.Semaphore()
    training_seg_pin_semaphore = threading.Semaphore()
    validation_pin_semaphore = threading.Semaphore()
    validation_seg_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()
    training_seg_pin_semaphore.acquire()
    validation_pin_semaphore.acquire()
    validation_seg_pin_semaphore.acquire()

    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()

    training_seg_pin_args = (training_seg_queue, pinned_training_seg_queue, training_seg_pin_semaphore)
    training_seg_pin_thread = threading.Thread(target=pin_memory, args=training_seg_pin_args)
    training_seg_pin_thread.daemon = True
    training_seg_pin_thread.start()

    validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True
    validation_pin_thread.start()

    validation_seg_pin_args = (validation_seg_queue, pinned_validation_seg_queue, validation_seg_pin_semaphore)
    validation_seg_pin_thread = threading.Thread(target=pin_memory, args=validation_seg_pin_args)
    validation_seg_pin_thread.daemon = True
    validation_seg_pin_thread.start()

    print("building model...")
    nnet = NetworkFactory(flag=True)

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)

    if start_iter:
        nnet.load_params(start_iter)
        nnet.set_lane_lr(lane_lr)
        print("training starts from iteration {}".format(start_iter + 1))
    else:
        nnet.set_lane_lr(lane_lr)
        nnet.set_seg_lr(seg_lr)

    print("training start...")
    nnet.cuda()
    nnet.train_mode()
    # summary(nnet.train, (3, 512, 512))
    header = None
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    writer = SummaryWriter(log_dir=os.path.join('runs', system_configs.exp_name))
    scaler = GradScaler()

    with stdout_to_tqdm() as save_stdout:
        for iteration in metric_logger.log_every(tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=67), print_freq=10, header=header):
            training = pinned_training_queue.get(block=True)
            training_seg = pinned_training_seg_queue.get(block=True)
            
            viz_split = 'train'
            model_mode = 'seg'
            # lr scheduler
            lane_lr = system_configs.learning_rate*(1 - iteration/max_iteration)**0.9
            seg_lr = system_configs.seg_learning_rate*(1 - iteration/max_iteration)**0.9
            nnet.set_lane_lr(lane_lr)
            nnet.set_seg_lr(seg_lr)
            
            save = True if (display and iteration % display == 0) else False
            seg_loss = nnet.train(model_mode, iteration, scaler, save, viz_split, **training_seg)

            if iteration % snapshot == 0:
                nnet.save_params(iteration)

            model_mode = 'lane_shape'
            save = True if (display and iteration % display == 0) else False
            
            (set_lane_loss, lane_loss_dict) = nnet.train(model_mode, iteration, scaler, save, viz_split, **training)
            (lane_loss_dict_reduced, lane_loss_dict_reduced_unscaled, lane_loss_dict_reduced_scaled, lane_loss_value) = lane_loss_dict

            writer.add_scalar('train_loss/train_lane_loss', lane_loss_value, iteration)
            writer.add_scalar('train_loss/train_seg_loss', seg_loss, iteration)
            for loss, lane_loss_value in lane_loss_dict_reduced.items():
                writer.add_scalar('train_loss_reduced/{}'.format(loss), lane_loss_value, iteration)
            for loss, lane_loss_value in lane_loss_dict_reduced_unscaled.items():
                writer.add_scalar('train_loss_unscaled/{}'.format(loss), lane_loss_value, iteration)
            for loss, lane_loss_value in lane_loss_dict_reduced_scaled.items():
                writer.add_scalar('train_loss_scaled/{}'.format(loss), lane_loss_value, iteration)
            writer.add_scalar('lane_lr', lane_lr, iteration)
            writer.add_scalar('seg_lr', seg_lr, iteration)
            
            metric_logger.update(seg_loss=seg_loss, lane_loss=lane_loss_value)
            metric_logger.update(class_error=lane_loss_dict_reduced['class_error'])
            metric_logger.update(lr=lane_lr)
            

            del set_lane_loss

            if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
                nnet.eval_mode()
                viz_split = 'val'

                model_mode = 'seg'
                save = True
                validation_seg = pinned_validation_seg_queue.get(block=True)
                val_seg_loss = nnet.validate(model_mode, iteration, save, viz_split, **validation_seg)

                model_mode = 'lane_shape'
                save = True
                validation = pinned_validation_queue.get(block=True)
                (val_set_loss, val_loss_dict) = nnet.validate(model_mode, iteration, save, viz_split, **validation)
                (lane_loss_dict_reduced, lane_loss_dict_reduced_unscaled, lane_loss_dict_reduced_scaled, lane_loss_value) = val_loss_dict

                writer.add_scalar('val_loss/val_lane_loss', lane_loss_value, iteration)
                writer.add_scalar('val_loss/val_seg_loss', val_seg_loss, iteration)
                for loss, lane_loss_value in lane_loss_dict_reduced.items():
                    writer.add_scalar('val_loss_reduced/{}'.format(loss), lane_loss_value, iteration)
                for loss, lane_loss_value in lane_loss_dict_reduced_unscaled.items():
                    writer.add_scalar('val_loss_unscaled/{}'.format(loss), lane_loss_value, iteration)
                for loss, lane_loss_value in lane_loss_dict_reduced_scaled.items():
                    writer.add_scalar('val_loss_scaled/{}'.format(loss), lane_loss_value, iteration)

                print('[VAL LOG]\t[Saving training and evaluating images...]')
                metric_logger.update(seg_loss=val_seg_loss, lane_loss=lane_loss_value)
                metric_logger.update(class_error=lane_loss_dict_reduced['class_error'])
                metric_logger.update(lr=lane_lr)
                nnet.train_mode()

            if iteration % snapshot == 0:
                nnet.save_params(iteration+1)

            if iteration % stepsize == 0:
                lane_lr /= decay_rate
                nnet.set_lane_lr(lane_lr)

            if iteration % (training_size // batch_size) == 0:
                metric_logger.synchronize_between_processes()
                print("Averaged stats:", metric_logger)


    # sending signal to kill the thread
    training_pin_semaphore.release()
    validation_pin_semaphore.release()

    # terminating data fetching processes
    for training_task in training_tasks:
        training_task.terminate()
    for training_seg_task in training_seg_tasks:
        training_seg_task.terminate()
    for validation_task in validation_tasks:
        validation_task.terminate()
    for validation_seg_task in validation_seg_tasks:
        validation_seg_task.terminate()

if __name__ == "__main__":
    args = parse_args()

    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)

    configs["system"]["snapshot_name"] = args.model_name
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split   = system_configs.val_split

    threads = args.threads  # 4 every 4 epoch shuffle the indices
    print("using {} threads".format(threads))

    dataset = system_configs.dataset
    print("loading all datasets {}...".format(dataset))
    training_dbs  = [datasets[dataset](configs["db"], train_split) for _ in range(threads)]
    # print('training_dbs', training_dbs[0])
    validation_db = datasets[dataset](configs["db"], val_split)
    print("len of training db: {}".format(len(training_dbs[0].db_inds)))
    print("len of testing db: {}".format(len(validation_db.db_inds)))

    print('-------------------seg dataset-------------------')
    seg_dataset = system_configs.seg_dataset
    seg_train_split = system_configs.seg_train_split
    seg_val_split = system_configs.seg_val_split
    seg_csv_path = system_configs.seg_csv_path
    print("loading all seg datasets {}...".format(seg_dataset))
    training_seg_dbs = [datasets[seg_dataset](configs["db"], seg_train_split, seg_csv_path) for _ in range(threads)]
    validation_seg_db = datasets[seg_dataset](configs["db"], seg_val_split, seg_csv_path)
    print(f'len of training seg db: {len(training_seg_dbs[0].db_inds)}')
    print(f'len of testing seg db: {len(validation_seg_db.db_inds)}')

    print("freeze the pretrained network: {}".format(args.freeze))
    train(training_dbs, validation_db, training_seg_dbs, validation_seg_db, args.start_iter, args.freeze) # 0
