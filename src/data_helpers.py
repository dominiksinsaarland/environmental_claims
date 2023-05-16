import json
import os
from sklearn.model_selection import StratifiedKFold
import numpy as np

def get_dataset_splits():

	train_fn = "data/train.jsonl"
	dev_fn = "data/dev.jsonl"
	test_fn = "data/test.jsonl"

	with open(train_fn) as f:
		data = [json.loads(i) for i in f]
		X_train = [i["text"] for i in data]
		y_train = [i["label"] for i in data]
	with open(dev_fn) as f:
		data = [json.loads(i) for i in f]
		X_validation = [i["text"] for i in data]
		y_validation = [i["label"] for i in data]
	with open(test_fn) as f:
		data = [json.loads(i) for i in f]
		X_test = [i["text"] for i in data]
		y_test = [i["label"] for i in data]
	return X_train, y_train, X_validation, y_validation, X_test, y_test


def get_cv_splits():
	X_train, y_train, X_validation, y_validation, X_test, y_test = get_dataset_splits()
	X = X_train + X_validation + X_test
	y = y_train + y_validation + y_test
	
	X, y = np.array(X), np.array(y)
	skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
	skf.get_n_splits(X, y)
	for i, (train_index, test_index) in enumerate(skf.split(X, y)):
		X_train, y_train = X[train_index], y[train_index]
		X_test, y_test = X[test_index], y[test_index]
		yield X_train, y_train, X_test, y_test

def round_float(number):
	return str(number.round(3) * 100)

