#!/usr/bin/python3

import cv2 as cv
import numpy as np
import time

def sample_a_batch(images_list, batch_size):
    mini_batch_file_list = np.random.choice(images_list, batch_size)

    load_size = 286
    fine_size = 256
    input_tensor = np.zeros([batch_size,fine_size,fine_size,3])
    for k in range(batch_size):
        # read and resize image to image_size, then cast to float32
        image_int =  cv.resize(cv.imread(mini_batch_file_list[k]),(load_size,load_size) )
        xy_offset = np.random.random_integers(0,29)
        image_int = image_int[ xy_offset:xy_offset+256, xy_offset:xy_offset+256, :]

        p = np.random.rand()
        if p > 0.5:
            image_int = np.fliplr(image_int)

        image_np = np.float32( image_int )
        # deterministic normalization
        image_np = image_np/127.5 - 1.0
        input_tensor[k] = image_np

    return input_tensor


def save_model_params():
    None

def restore_model_params():
    None
