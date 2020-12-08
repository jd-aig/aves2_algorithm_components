from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.externals import joblib
import pandas as pd
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../data/")
parser.add_argument("--output_path", type=str, default="./output/")
parser.add_argument("--target", type=str, default="MEDV")
parser.add_argument("--loss", type=str, default="linear")#square,exponential
parser.add_argument("--n_estimators", type=int, default=50)
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

model = AdaBoostRegressor(loss=args.loss,n_estimators=args.n_estimators)
model.fit(x_train,y_train)

save_path = os.path.join(args.output_path,'model.m')
joblib.dump(model,save_path)
print("AdaBoostRegressor train finished.save model in model/output/model.m")
