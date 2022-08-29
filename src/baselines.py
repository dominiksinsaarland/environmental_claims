

# let's just train SVM

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import cross_val_score


from data_helpers import get_dataset_splits, round_float
import numpy as np
X_train, y_train, X_validation, y_validation, X_test, y_test = get_dataset_splits()

def create_table_1(fn="claims_final.json"):
	X_train, y_train, X_validation, y_validation, X_test, y_test = get_dataset_splits()
	for split in ["train", "dev", "test", "all"]:
		out_str = split + " & "
		if split == "train":
			out_str += str(len(X_train)) + " & "
			out_str += str(np.mean([len(i.split()) for i in X_train]).round(1)) + " & " 
			#out_str += str(np.median([len(i.split()) for i in X_train]).round(1)) + " & " 
			out_str += str(np.mean(y_train).round(2))

		if split == "dev":
			out_str += str(len(X_validation)) + " & "
			out_str += str(np.mean([len(i.split()) for i in X_validation]).round(1)) + " & " 
			#out_str += str(np.median([len(i.split()) for i in X_validation]).round(1)) + " & " 
			out_str += str(np.mean(y_validation).round(2))

		if split == "test":
			out_str += str(len(X_test)) + " & "
			out_str += str(np.mean([len(i.split()) for i in X_test]).round(1)) + " & " 
			#out_str += str(np.median([len(i.split()) for i in X_test]).round(1)) + " & " 
			out_str += str(np.mean(y_test).round(2))
		if split == "all":
			X_all = X_train + X_validation + X_test
			y_all = y_train + y_validation + y_test
			out_str += str(len(X_all)) + " & "
			out_str += str(np.mean([len(i.split()) for i in X_all]).round(1)) + " & " 
			#out_str += str(np.median([len(i.split()) for i in X_all]).round(1)) + " & " 
			out_str += str(np.mean(y_all).round(2))
		out_str += r" \\ "
		print (out_str)



def baselines():
	# majority
	out_str = "majority & "
	all_preds, all_labels = [], []
	
	X_train, y_train, X_validation, y_validation, X_test, y_test = get_dataset_splits()
	preds = [0] * len(y_validation)

	macro = round_float(metrics.f1_score(y_validation, preds, average="macro", zero_division=0))
	micro = round_float(metrics.f1_score(y_validation, preds, average="micro", zero_division=0))
	out_str += micro + " & " + macro + " & "

	preds = [0] * len(y_test)

	macro = round_float(metrics.f1_score(y_test, preds, average="macro", zero_division=0))
	micro = round_float(metrics.f1_score(y_test, preds, average="micro", zero_division=0))
	out_str += micro + " & " + macro + r" \\"
	print (out_str) 

	# random
	out_str = "random & "
	
	X_train, y_train, X_validation, y_validation, X_test, y_test = get_dataset_splits()
	preds = np.random.randint(0,2,size=len(y_validation),dtype=int)

	macro = round_float(metrics.f1_score(y_validation, preds, average="macro", zero_division=0))
	micro = round_float(metrics.f1_score(y_validation, preds, average="micro", zero_division=0))
	out_str += micro + " & " + macro + " & "

	preds = np.random.randint(0,2,size=len(y_test),dtype=int)

	macro = round_float(metrics.f1_score(y_test, preds, average="macro", zero_division=0))
	micro = round_float(metrics.f1_score(y_test, preds, average="micro", zero_division=0))
	out_str += micro + " & " + macro + r" \\"
	print (out_str) 

baselines()

def tf_idf_baseline():
	classifier = LinearSVC(max_iter=50000)
	parameters = {
	'vect__max_features': [10000, 20000, 40000],
	'clf__C': [0.1, 1, 10],
	'clf__loss': ('hinge', 'squared_hinge')
	}

	out_str = "TF-IDF SVM & "

	# dev
	X_train, y_train, X_validation, y_validation, X_test, y_test = get_dataset_splits()

	text_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1, 3), min_df=5)), ('tfidf', TfidfTransformer()), ('clf', classifier),])
	text_clf.fit(X_train, y_train)
	preds = text_clf.predict(X_validation)

	macro = round_float(metrics.f1_score(y_validation, preds, average="macro", zero_division=0))
	micro = round_float(metrics.f1_score(y_validation, preds, average="micro", zero_division=0))
	out_str += micro + " & " + macro + " & "

	# test
	text_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1, 3), min_df=5)), ('tfidf', TfidfTransformer()), ('clf', classifier),])
	text_clf.fit(X_train, y_train)
	preds = text_clf.predict(X_test)
	macro = round_float(metrics.f1_score(y_test, preds, average="macro", zero_division=0))
	micro = round_float(metrics.f1_score(y_test, preds, average="micro", zero_division=0))
	out_str += micro + " & " + macro + r" \\"
	print (out_str) 

tf_idf_baseline()
