import json

def get_dataset_splits():
	with open("data/train.jsonl") as f:
		data = [json.loads(i) for i in f]
		X_train = [i["text"] for i in data]
		y_train = [i["label"] for i in data]

	with open("data/dev.jsonl") as f:
		data = [json.loads(i) for i in f]
		X_validation = [i["text"] for i in data]
		y_validation = [i["label"] for i in data]

	with open("data/test.jsonl") as f:
		data = [json.loads(i) for i in f]
		X_test = [i["text"] for i in data]
		y_test = [i["label"] for i in data]
	return X_train, y_train, X_validation, y_validation, X_test, y_test


def round_float(number):
	return str(number.round(3) * 100)

