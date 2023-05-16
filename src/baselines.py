

# let's just train SVM

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import cross_val_score


from data_helpers import get_dataset_splits, round_float, get_cv_splits
import numpy as np
X_train, y_train, X_validation, y_validation, X_test, y_test = get_dataset_splits()

from scipy.stats import sem

def evaluate(gold, predictions):
	# acc, pr, rc, f1
	pr = round_float(metrics.precision_score(gold, predictions))
	rc = round_float(metrics.recall_score(gold, predictions))
	f1 = round_float(metrics.f1_score(gold, predictions))
	acc = round_float(metrics.accuracy_score(gold, predictions))
	return " & ".join((pr, rc, f1, acc))


def evaluate_all(gold, preds):
	pr, rc, f1, acc = [], [], [], []
	for g,p in zip(gold, preds):
		pr.append(metrics.precision_score(g, p))
		rc.append(metrics.recall_score(g, p))
		f1.append(metrics.f1_score(g, p))
		acc.append(metrics.accuracy_score(g, p))
	
	out_str = round_float(np.mean(pr)) + r" \pm " + str(sem(pr).round(3) * 100) + " & " + round_float(np.mean(rc)) + r" \pm " + str(sem(rc).round(3) * 100) + " & " + round_float(np.mean(f1)) + r" \pm " + str(sem(f1).round(3) * 100) + " & " + round_float(np.mean(acc)) + r" \pm " + str(sem(acc).round(3) * 100)
	return out_str
def create_table_1():
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
		out_str += r" \\ \hline"
		print (out_str)
		
create_table_1()


def baselines():
	# majority
	out_str = "majority & "
	all_preds, all_labels = [], []

	# crossvalidation...
	for X_train, y_train, X_test, y_test in get_cv_splits():
		all_labels.extend(y_test)
		preds = [0] * len(y_test)
		all_preds.extend(preds)
		
	out_str += evaluate(all_labels, all_preds) + " & "			
	
	# dev	
	X_train, y_train, X_validation, y_validation, X_test, y_test = get_dataset_splits()
	preds = [0] * len(y_validation)
	out_str += evaluate(y_validation, preds) + " & "
	
	# test
	preds = [0] * len(y_test)
	out_str += evaluate(y_test, preds) + r" \\"

	print (out_str) 

	# random
	out_str = "random & "

	all_preds, all_labels = [], []
	for X_train, y_train, X_test, y_test in get_cv_splits():
		all_labels.extend(y_test)
		preds = np.random.randint(0,2,size=len(y_test),dtype=int)
		all_preds.extend(preds)
		
	out_str += evaluate(all_labels, all_preds) + " & "			


	
	# dev
	preds = np.random.randint(0,2,size=len(y_validation),dtype=int)
	out_str += evaluate(y_validation, preds) + " & "

	# test
	preds = np.random.randint(0,2,size=len(y_test),dtype=int)
	out_str += evaluate(y_test, preds) + r" \\"

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

	# CV
	all_preds, all_labels = [], []
	for X_train, y_train, X_test, y_test in get_cv_splits():
		text_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1, 3), min_df=5)), ('tfidf', TfidfTransformer()), ('clf', classifier),])
		text_clf.fit(X_train, y_train)
		all_preds.extend(text_clf.predict(X_test))
		all_labels.extend(y_test)

	out_str += evaluate(all_labels, all_preds) + " & "
	
	X_train, y_train, X_validation, y_validation, X_test, y_test = get_dataset_splits()

	# dev
	text_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1, 3), min_df=5)), ('tfidf', TfidfTransformer()), ('clf', classifier),])
	text_clf.fit(X_train, y_train)
	preds = text_clf.predict(X_validation)
	
	out_str += evaluate(y_validation, preds) + " & "
	# test
	preds = text_clf.predict(X_test)
	out_str += evaluate(y_test, preds) + " & "
	print (out_str)		
tf_idf_baseline()


def character_n_gram_baseline():
	classifier = LinearSVC(max_iter=50000)
	parameters = {
	'vect__max_features': [10000, 20000, 40000],
	'clf__C': [0.1, 1, 10],
	'clf__loss': ('hinge', 'squared_hinge')
	}

	out_str = "Character n-gram SVM & "

	# CV
	# vectorizer = CountVectorizer(ngram_range=(1, 10), token_pattern = r"(?u)\b\w+\b", analyzer='char')

	all_preds, all_labels = [], []
	for X_train, y_train, X_test, y_test in get_cv_splits():
		# text_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1, 3), min_df=5)), ('tfidf', TfidfTransformer()), ('clf', classifier),])
		
		text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 10), token_pattern = r"(?u)\b\w+\b", analyzer='char', min_df=5)), ('tfidf', TfidfTransformer()), ('clf', classifier),])
		
		
		
		
		text_clf.fit(X_train, y_train)
		all_preds.extend(text_clf.predict(X_test))
		all_labels.extend(y_test)

	out_str += evaluate(all_labels, all_preds) + " & "
	
	X_train, y_train, X_validation, y_validation, X_test, y_test = get_dataset_splits()

	# dev
	text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 10), token_pattern = r"(?u)\b\w+\b", analyzer='char', min_df=10)), ('tfidf', TfidfTransformer()), ('clf', classifier),])
	text_clf.fit(X_train, y_train)
	preds = text_clf.predict(X_validation)
	
	out_str += evaluate(y_validation, preds) + " & "
	# test
	preds = text_clf.predict(X_test)
	out_str += evaluate(y_test, preds) + " & "
	print (out_str)
	
character_n_gram_baseline()
