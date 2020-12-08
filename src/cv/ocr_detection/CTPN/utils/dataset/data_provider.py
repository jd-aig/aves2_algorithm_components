# encoding:utf-8
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.dataset.data_util import GeneratorEnqueuer

DATA_FOLDER = "data/dataset/mlt_new/"


def get_training_data(data_folder):
    #print(os.path.join(DATA_FOLDER, "image"))
    print(os.path.join(data_folder, "image"))
    img_files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    #for parent, dirnames, filenames in os.walk(os.path.join(DATA_FOLDER, "image")):
    for parent, dirnames, filenames in os.walk(os.path.join(data_folder, "image")):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    img_files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(img_files)))
    return img_files


def load_annoataion(p):
    bbox = []
    with open(p, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(",")
        x_min, y_min, x_max, y_max = map(int, line)
        bbox.append([x_min, y_min, x_max, y_max, 1])
    return bbox


def generator(data_folder, shuffle = True, vis=False):
    print('generator')
    #image_list = np.array(get_training_data())
    #print('{} training images in {}'.format(image_list.shape[0], DATA_FOLDER))
    image_list = np.array(get_training_data(data_folder))
    print('{} training images in {}'.format(image_list.shape[0], data_folder))

    index = np.arange(0, image_list.shape[0])
    if shuffle:
        while True:
            np.random.shuffle(index)
            for i in index:
                try:
                    im_fn = image_list[i]
                    im = cv2.imread(im_fn)
                    h, w, c = im.shape
                    im_info = np.array([h, w, c]).reshape([1, 3])
                    
                    _, fn = os.path.split(im_fn)
                    fn, _ = os.path.splitext(fn)
                    txt_fn = os.path.join(data_folder, "label", fn + '.txt')
                    if not os.path.exists(txt_fn):
                        print("Ground truth for image {} not exist!".format(im_fn))
                        continue
                    bbox = load_annoataion(txt_fn)
                    if len(bbox) == 0:
                        print("Ground truth for image {} empty!".format(im_fn))
                        continue
                    yield [im], bbox, im_info
                except Exception as e:
                    print(e)
                    continue
    else:
        for i in index:
            try:
                im_fn = image_list[i]
                im = cv2.imread(im_fn)
                h, w, c = im.shape
                im_info = np.array([h, w, c]).reshape([1, 3])

                _, fn = os.path.split(im_fn)
                fn, _ = os.path.splitext(fn)
                txt_fn = os.path.join(data_folder, "label", fn + '.txt')
                if not os.path.exists(txt_fn):
                    print("Ground truth for image {} not exist!".format(im_fn))
                    continue
                bbox = load_annoataion(txt_fn)
                if len(bbox) == 0:
                    print("Ground truth for image {} empty!".format(im_fn))
                    continue
                yield [im], bbox, im_info
            except Exception as e:
                print(e)
                continue

def get_batch(num_workers, data_folder, **kwargs):
    print('get batch')
    try:
        enqueuer = GeneratorEnqueuer(generator(data_folder, **kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()

'''
if __name__ == '__main__':
    gen = get_batch(num_workers=2, vis=True)
    while True:
        image, bbox, im_info = next(gen)
        print('done')
'''
