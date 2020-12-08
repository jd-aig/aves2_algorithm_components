from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.externals import joblib
import pandas as pd
import argparse
import os
import json
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../data/")
parser.add_argument("--model_dir", type=str, default="./model/")
parser.add_argument("--output_path", type=str, default="./output/")
parser.add_argument("--target", type=str, default="virginica")
args = parser.parse_args()

val_dataset = os.path.join(args.data_dir,'val.csv')
val_data = pd.read_csv(val_dataset)
lst = val_data.columns.values.tolist()
idx = lst.index(args.target)
del lst[idx]

y_val = val_data.ix[:,args.target].values
x_val = val_data.ix[:,lst].values

save_path = os.path.join(args.model_dir,'model.m')
model = joblib.load(save_path)
predict = model.predict(x_val)

c_m = confusion_matrix(y_val, predict)
c_m = str(c_m)
print('confusion_matrix : \n', c_m)
c_m = c_m.replace("[", "")
c_m = c_m.replace("]", "")

fcm = open(os.path.join(args.output_path,"confusion_matrix.txt"), "w")
cm_lines = c_m.split("\n")
for i in range(len(cm_lines)):
   cm = str(cm_lines[i])
   cm = cm.lstrip()
   cm = cm.rstrip()
   cm = cm.split(" ")
   for j in range(len(cm)):
       num = str(cm[j])
       num = num.lstrip()
       num = num.rstrip()
       if not num.isspace() and num != '':
           fcm.write(str(cm[j]))
           if j < (len(cm)-1):
               fcm.write("\t")
   fcm.write("\n")
fcm.close()

accuracy = accuracy_score(y_val, predict)
p_score = precision_score(y_val, predict , average='macro')
r_score = recall_score(y_val, predict, average='macro')
f1 = f1_score(y_val, predict, average='macro')

json_dict = {}
json_dict["accuracy"] = accuracy
json_dict["p_score"] = p_score
json_dict["r_score"] = r_score
json_dict["f1_score"] = f1
json_data = json.dumps(json_dict)
f = open(os.path.join(args.output_path,"result.json"),"w")
f.write(str(json_data))
f.close()

print('accuracy : ', accuracy)
print('p_score : ', p_score)
print('r_score : ', r_score)
print('f1_score : ', f1)
