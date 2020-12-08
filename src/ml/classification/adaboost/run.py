from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import argparse
import os
import json
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../data/")
parser.add_argument("--output_path", type=str, default="./output/")
parser.add_argument("--target", type=str, default="virginica")
parser.add_argument("--algorithm", type=str, default="SAMME")#SAMME,SAMME.R
parser.add_argument("--n_estimators", type=int, default=50)
args = parser.parse_args()

train_dataset = os.path.join(args.data_dir,'train.csv')
train_data = pd.read_csv(train_dataset)

lst = train_data.columns.values.tolist()
idx = lst.index(args.target)
del lst[idx]

y_train = train_data.ix[:,args.target].values
x_train = train_data.ix[:,lst].values

model = AdaBoostClassifier(algorithm=args.algorithm,n_estimators=args.n_estimators)
model.fit(x_train,y_train)

save_path = os.path.join(args.output_path,'model.m')
joblib.dump(model,save_path)
