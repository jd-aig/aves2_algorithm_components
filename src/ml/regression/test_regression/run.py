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

test_dataset = None
if os.path.exists(os.path.join(args.data_dir,'test.csv')):
  test_dataset = os.path.join(args.data_dir,'test.csv')
elif os.path.exists(os.path.join(args.data_dir,'val.csv')):
  test_dataset = os.path.join(args.data_dir,'val.csv')
elif os.path.exists(os.path.join(args.data_dir,'train.csv')):
  test_dataset = os.path.join(args.data_dir,'train.csv')
else:
  print("ERROR:test file invalid!")
  exit()
test_data = pd.read_csv(test_dataset)
lst = test_data.columns.values.tolist()
idx = lst.index(args.target)
del lst[idx]

y_val = test_data.ix[:,args.target].values
x_val = test_data.ix[:,lst].values

model_path = os.path.join(args.model_dir,'model.m')
if not os.path.exists(model_path):
  print("ERROR: model.m is not exists")
  exit()
model = joblib.load(model_path)
predict = model.predict(x_val)
pred_csv = pd.concat([test_data,pd.DataFrame(columns=['PREDICT'],data=predict)],sort=False,axis=1)
pred_csv.to_csv(os.path.join(args.output_path,'result.csv'),float_format = '%.3f')
print("Predict successful! results in data/output/result.csv")
