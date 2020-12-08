from sklearn.mixture import GaussianMixture 
from sklearn.metrics import calinski_harabaz_score 
from sklearn.externals import joblib
import pandas as pd
import argparse
import os
import json
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../data/")
parser.add_argument("--output_path", type=str, default="./output/")
parser.add_argument("--n_components", type=int, default=2)
parser.add_argument("--covariance_type", type=str, default='full')#full,tied,diag,spherical
parser.add_argument("--tol", type=float, default=0.001)
parser.add_argument("--max_iter", type=int, default=100)
args = parser.parse_args()

dataset = os.path.join(args.data_dir,'data.csv')
data = pd.read_csv(dataset)

x = data.ix[:,:].values

model = GaussianMixture(n_components=args.n_components, covariance_type=args.covariance_type, tol=args.tol, max_iter=args.max_iter)
model.fit(x)
y_pred = model.predict(x)

save_path = os.path.join(args.output_path,'model.m')
joblib.dump(model,save_path)

c_h_score = calinski_harabaz_score(x,y_pred)

json_dict = {}
json_dict["calinski_harabaz_score"] = c_h_score 
json_data = json.dumps(json_dict)
f = open(os.path.join(args.output_path,"result.json"),"w")
f.write(str(json_data))
f.close()

pred_csv = pd.concat([data,pd.DataFrame(columns=['PREDICT'],data=y_pred.tolist())],sort=False,axis=1)
pred_csv.to_csv(os.path.join(args.output_path,'result.csv'),float_format = '%.5f')

print('calinski_harabaz_score : ', c_h_score)
