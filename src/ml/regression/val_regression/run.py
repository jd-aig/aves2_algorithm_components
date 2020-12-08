from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import argparse
import os
import json
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../data/")
parser.add_argument("--model_dir", type=str, default="./model/")
parser.add_argument("--output_path", type=str, default="./output/")
parser.add_argument("--target", type=str, default="MEDV")
args = parser.parse_args()

val_dataset = os.path.join(args.data_dir,'val.csv')
if not os.path.exists(val_dataset):
  print("ERROR: val.csv is not exists!")
  exit()
val_data = pd.read_csv(val_dataset)

lst = val_data.columns.values.tolist()
idx = lst.index(args.target)
del lst[idx]

y_val = val_data.ix[:,args.target].values
x_val = val_data.ix[:,lst].values

model_path = os.path.join(args.model_dir,'model.m')
if not os.path.exists(model_path):
  print("ERROR: model.m is not exists")
  exit()
model = joblib.load(model_path)
predict = model.predict(x_val)

rmse = (np.sqrt(mean_squared_error(y_val, predict)))
r2 = r2_score(y_val,predict)

json_dict = {}
json_dict["rmse"] = rmse
json_dict["r2_score"] = r2
json_data = json.dumps(json_dict)
f = open(os.path.join(args.output_path,"result.json"),"w")
f.write(str(json_data))
f.close()
print('rmse : ', rmse)
print('r2_score : ', r2)
print('val successful! results in data/output/result.json')
