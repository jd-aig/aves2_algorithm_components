from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.externals import joblib
import pandas as pd
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../data/")
parser.add_argument("--output_path", type=str, default="./output/")
parser.add_argument("--target", type=str, default="MEDV")
parser.add_argument("--fit_intercept", type=bool, default=True)
parser.add_argument("--normalize", type=bool, default=False)
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

model = LinearRegression(fit_intercept=args.fit_intercept,normalize=args.normalize,copy_X=True,n_jobs=1)
model.fit(x_train,y_train,sample_weight=None)

save_path = os.path.join(args.output_path,'model.m')
joblib.dump(model,save_path)
print("LinearRegression train finished.save model in model/output/model.m")
