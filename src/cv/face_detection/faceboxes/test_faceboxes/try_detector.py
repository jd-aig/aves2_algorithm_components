import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from PIL import Image, ImageDraw
import os
import cv2
import time

from face_detector import FaceDetector
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",type=str,default='../data_dir/')
parser.add_argument("--model_dir",type=str,default='../model_dir/')
parser.add_argument("--output_path",type=str,default='../output_path/')
args = parser.parse_args()

MODEL_PATH = os.path.join(args.model_dir,'frozen_inference_graph.pb')
face_detector = FaceDetector(MODEL_PATH, gpu_memory_fraction=0.95, visible_device_list='0')

def draw_boxes_on_image(image, boxes, scores):

    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy, 'RGBA')
    width, height = image.size

    for b, s in zip(boxes, scores):
        ymin, xmin, ymax, xmax = b
        fill = (255, 0, 0, 45)
        outline = 'red'
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            fill=fill, outline=outline
        )
        draw.text((xmin, ymin), text='{:.3f}'.format(s))
    return image_copy

times = []
#print(os.listdir(args.data_dir))
for filename in os.listdir(os.path.join(args.data_dir,'images')):
    path = os.path.join(args.data_dir,'images',filename)
    image_array = cv2.imread(path)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    #image = Image.fromarray(image_array)
    start = time.time()
    boxes, scores = face_detector(image_array, score_threshold=0.3)
    print("image cost time=%.2f ms" %(1000*(time.time()-start)))
    image_out = draw_boxes_on_image(Image.fromarray(image_array), boxes, scores)
    image_out.save(os.path.join(args.output_path,filename),quality=95) 
    #print(boxes)
    #print(scores)
