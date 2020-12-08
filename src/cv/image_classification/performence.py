import sklearn
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import json
import argparse
import numpy

def per(labels,logits,files,output_dir):
	
	preds = [i.index(max(i)) for i in logits]
	num_class = len(logits[0])
	enc = OneHotEncoder(dtype=numpy.int)

	X = [[i] for i in range(num_class)]
	labelsll = [[i] for i in labels]
	enc.fit(X)
	labels_one_hot = enc.transform(labelsll).toarray()

	classif_per = {}
	accuracy_score = metrics.accuracy_score(labels,preds)
	classif_per['accuracy'] = accuracy_score
	
	precision_score = metrics.precision_score(labels,preds,average='macro')
	classif_per['precision'] = precision_score

	f1_score = metrics.f1_score(labels,preds,average='macro')
	classif_per['f1_score'] = f1_score

	recall_score = metrics.recall_score(labels,preds,average='macro')
	classif_per['recall_score'] = recall_score
	
	roc_auc_score_micro = metrics.roc_auc_score(labels_one_hot, logits, average='micro')
	classif_per['auc_micro'] = roc_auc_score_micro	

	classification_report_dict = {}
	classification_report = metrics.classification_report(labels,preds,labels=range(num_class))
	classification_report = str(classification_report).split('\n')

	for i in range(len(classification_report)):
		x = classification_report[i]
		x = str(x).split(' ')
		xx =[]
		for j in x:
			try:
				assert len(j)>0
				xx.append(j)
			except:
				continue
		if len(xx) == 4:
			classification_report_dict['evaluation_index'] = xx
		elif len(xx) == 7:
			classification_report_dict['avg_all'] = xx[3:]
		elif len(xx)>0:
			classification_report_dict[xx[0]]=xx[1:]

	classif_per['classification_report'] = classification_report_dict
	confusion_matrix = metrics.confusion_matrix(labels,preds)
	confusion_matrix_str = ''
	for i in confusion_matrix:
		for j in i:
			confusion_matrix_str = confusion_matrix_str + str(j) + '\t'
		confusion_matrix_str = confusion_matrix_str + '\n'

	classif_per_path = output_dir + 'result.json'
	
	jsObj = json.dumps(classif_per)
	fileObject = open(classif_per_path, 'w')
	fileObject.write(jsObj)
	fileObject.close()

	confusion_matrix_path = output_dir + 'confusion_matrix.txt'
	confusion_matrix_file = open(confusion_matrix_path,'w')
	confusion_matrix_file.write(confusion_matrix_str)
	confusion_matrix_file.close()

	correct_path =  output_dir + 'correct.txt'
	error_path = output_dir + 'error.txt'
	right = open(correct_path,'w')
	error = open(error_path,'w')
	for i in range(len(files)):
		f = files[i]
		l = labels[i]
		p = preds[i]
		if l == p:
			right.write(f + '\t' + str(l) + '\t' + str(p) + '\n')
		else:
			error.write(f + '\t' + str(l) + '\t' + str(p) + '\n')
	right.close()
	error.close()
	
