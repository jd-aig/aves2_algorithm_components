import os
import argparse
from object_detection.utils import label_map_util

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=24)
args = parser.parse_args()

label_map_path = os.path.join(args.data_dir, 'ImageSets', 'label_map.pbtxt')
label_map_dict = label_map_util.get_label_map_dict(label_map_path)
num_classes_val = len(label_map_dict)

train_txt = os.path.join(args.data_dir, 'ImageSets', 'train.txt')
val_txt = os.path.join(args.data_dir, 'ImageSets', 'val.txt')
count = 0
for index, line in enumerate(open(train_txt,'r')): 
    count += 1
num_examples_train = count
count = 0
for index, line in enumerate(open(val_txt,'r')):
    count += 1
num_examples_val = count

num_train_steps_val = num_examples_train//args.batch_size
if num_train_steps_val == 0:
    num_train_steps_val = 1
num_train_steps_val = num_train_steps_val*args.epochs
decay_steps_val = num_examples_train//args.batch_size
if decay_steps_val == 0:
    decay_steps_val = 1
decay_factor_val = 0.9
print("num_classes_val = ",num_classes_val," ,decay_steps_val = ",decay_steps_val," ,decay_factor_val = ",decay_factor_val," ,num_examples_val = ",num_examples_val," ,num_train_steps_val = ",num_train_steps_val)

f = open(os.path.join("cal_params.txt"),"w")
f.write(str(num_classes_val)+'\n')
f.write(str(decay_steps_val)+'\n')
f.write(str(decay_factor_val)+'\n')
f.write(str(num_examples_val)+'\n')
f.write(str(num_train_steps_val)+'\n')
f.close()
