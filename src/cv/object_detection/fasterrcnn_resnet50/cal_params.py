import os
import argparse
from object_detection.utils import label_map_util

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--learning_rate", type=float, default=0.0001)
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
decay_steps_val_1 = num_train_steps_val//3
if decay_steps_val_1 == 0:
    decay_steps_val_1 = 1
decay_steps_val_2 = num_train_steps_val//3*2
if decay_steps_val_2 == 0:
    decay_steps_val_2 = 1
if decay_steps_val_2 == decay_steps_val_1:
    decay_steps_val_2 = decay_steps_val_1+1
learning_rate_val_1 = args.learning_rate*0.1
learning_rate_val_2 = args.learning_rate*0.01

print("num_classes_val = ",num_classes_val," ,decay_steps_val_1 = ",decay_steps_val_1," ,decay_steps_val_2 = ",decay_steps_val_2," ,num_examples_val = ",num_examples_val," ,num_train_steps_val = ",num_train_steps_val," ,learning_rate_val_1 = ",learning_rate_val_1," ,learning_rate_val_2 = ",learning_rate_val_2)

f = open(os.path.join("cal_params.txt"),"w")
f.write(str(num_classes_val)+'\n')
f.write(str(decay_steps_val_1)+'\n')
f.write(str(decay_steps_val_2)+'\n')
f.write(str(num_examples_val)+'\n')
f.write(str(num_train_steps_val)+'\n')
f.write(str(learning_rate_val_1)+'\n')
f.write(str(learning_rate_val_2)+'\n')
f.close()
