import json
import random
import argparse
from sklearn.metrics import precision_recall_fscore_support, classification_report, f1_score
from sklearn.model_selection import train_test_split, KFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification, BigBirdConfig
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from data_helpers import get_dataset_splits, round_float, get_cv_splits
from sklearn import metrics
import numpy as np

def evaluate(gold, predictions):
	# acc, pr, rc, f1
	pr = round_float(metrics.precision_score(gold, predictions))
	rc = round_float(metrics.recall_score(gold, predictions))
	f1 = round_float(metrics.f1_score(gold, predictions))
	acc = round_float(metrics.accuracy_score(gold, predictions))
	return " & ".join((pr, rc, f1, acc))



class SequenceClassificationDataset(Dataset):
	def __init__(self, x, y, tokenizer):
		self.examples = list(zip(x,y))
		self.tokenizer = tokenizer
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

	def __len__(self):
		return len(self.examples)
	def __getitem__(self, idx):
		return self.examples[idx]
	def collate_fn(self, batch):
		model_inputs = self.tokenizer([i[0] for i in batch], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
		labels = torch.tensor([i[1] for i in batch]).to(self.device)
		return {"model_inputs": model_inputs, "label": labels}

def evaluate_epoch(model, dataset):
	model.eval()
	targets = []
	outputs = []
	probs = []
	with torch.no_grad():
		for batch in DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn):
			output = model(**batch["model_inputs"])
			logits = output.logits
			targets.extend(batch['label'].float().tolist())
			outputs.extend(logits.argmax(dim=1).tolist())
			probs.extend(logits.softmax(dim=1)[:,1].tolist())
	return targets, outputs, probs

def train_model(trainset, model_name):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	config = AutoConfig.from_pretrained(model_name)
	config.num_labels = 2
	config.gradient_checkpointing = True
	model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, cache_dir="../../transformer_models/").to(device)

	warmup_steps = 0
	train_dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=trainset.collate_fn)
	t_total = int(len(train_dataloader) * args.num_epochs / args.gradient_accumulation_steps)

	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
	{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
	{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	    ]
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

	model.zero_grad()
	optimizer.zero_grad()

	use_amp = True
	scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

	for epoch in range(args.num_epochs):
		model.train()
		t = tqdm(train_dataloader)
		# for i, batch in enumerate(train_dataloader):
		for i, batch in enumerate(t):
			with torch.cuda.amp.autocast(enabled=use_amp):
				output = model(**batch["model_inputs"], labels=batch['label'])
				loss = output.loss / args.gradient_accumulation_steps
			scaler.scale(loss).backward()

			if (i + 1) % args.gradient_accumulation_steps == 0:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				scaler.step(optimizer)
				scaler.update()
				scheduler.step()  # Update learning rate schedule
				optimizer.zero_grad()
	return model

def main(args):
	model_name = args.model_name
	try:
		tokenizer = AutoTokenizer.from_pretrained(model_name)
	except:
		tokenizer = AutoTokenizer.from_pretrained("roberta-base")


	out_str = os.path.basename(model_name) + " & "

	# cross-validation
	if args.do_cross_validation:
		X_train, y_train, X_validation, y_validation, X_test, y_test = get_dataset_splits()
		X = np.array(X_train + X_validation + X_test)
		y = np.array(y_train + y_validation + y_test)
		kf = KFold(n_splits=5, random_state=42, shuffle=True)	
		y_true, y_pred = [], []
		counter = 0
		for train_index, test_index in kf.split(X):
			print ("CV split ", counter)
			counter += 1
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			trainset = SequenceClassificationDataset(X_train, y_train, tokenizer)
			devset = SequenceClassificationDataset(X_test, y_test, tokenizer)

			model = train_model(trainset, model_name)

			targets, outputs, probs = evaluate_epoch(model, devset)
			y_true.extend(targets)
			y_pred.extend(outputs)
		out_str += evaluate(y_true, y_pred) + " & "
	X_train, y_train, X_validation, y_validation, X_test, y_test = get_dataset_splits()
	trainset = SequenceClassificationDataset(X_train, y_train, tokenizer)
	devset = SequenceClassificationDataset(X_validation, y_validation, tokenizer)
	model = train_model(trainset, model_name)

	# evaluate dev set	
	targets, outputs, probs = evaluate_epoch(model, devset)
	out_str += evaluate(targets, outputs) + " & "
	
	# evaluate test set
	devset = SequenceClassificationDataset(X_test, y_test, tokenizer)
	targets, outputs, probs = evaluate_epoch(model, devset)

	out_str += evaluate(targets, outputs) + r" \\ "

	print (out_str)
	return model, tokenizer


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--save_path', type=str, default="envclaim-distilroberta", help='Folder to save the weights')
	parser.add_argument('--model_name', type=str, default='distilroberta-base')
	parser.add_argument('--num_epochs', type=int, default=3)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
		        help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument("--batch_size", default=16, type=int,
		        help="Batch size per GPU/CPU for training.")
	parser.add_argument("--learning_rate", default=2e-5, type=float,
		        help="The initial learning rate for Adam.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float,
		        help="Epsilon for Adam optimizer.")
	parser.add_argument("--only_prediction", default=None, type=str,
		        help="Epsilon for Adam optimizer.")
	parser.add_argument('--do_save', action='store_true')
	parser.add_argument('--do_cross_validation', action='store_true')

	args = parser.parse_args()
	model, tokenizer = main(args)

	if args.do_save:
		model.save_pretrained(args.save_path)
		tokenizer.save_pretrained(args.save_path)

	# python src/transformer_models.py --do_save --save_path climatebert-environmental-claims --model_name climatebert/distilroberta-base-climate-f --do_cross_validation

