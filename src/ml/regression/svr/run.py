from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.externals import joblib
import pandas as pd
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../data/")
parser.add_argument("--output_path", type=str, default="./output/")
parser.add_argument("--target", type=str, default="MEDV")
parser.add_argument("--kernel", type=str, default="linear")#poly rbf
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--c_value", type=float, default=1.0)
parser.add_argument("--max_iter", type=int, default=100)
parser.add_argument("--degree", type=int, default=3)
args = parser.parse_args()

train_dataset = os.path.join(args.data_dir,'train.csv')
if not os.path.exists(train_dataset):
  print("ERROR: train.csv is not exists!")
  exit()
train_data = pd.read_csv(train_dataset)

lst = train_data.columns.values.tolist()
idx = lst.index(args.target)
del lst[idx]

y_train = train_data.ix[:,args.target].values
x_train = train_data.ix[:,lst].values

model = SVR(kernel=args.kernel,gamma=args.gamma, C=args.c_value,max_iter=args.max_iter,degree=args.degree)
model.fit(x_train,y_train)

save_path = os.path.join(args.output_path,'model.m')
joblib.dump(model,save_path)
print("SVR train finished.save model in model/output/model.m")
