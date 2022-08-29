from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader
import argparse
import json
import torch
from tqdm import tqdm

class SequenceClassificationDataset(Dataset):
	def __init__(self, x, tokenizer):
		self.examples = x
		self.tokenizer = tokenizer
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

	def __len__(self):
		return len(self.examples)
	def __getitem__(self, idx):
		return self.examples[idx]
	def collate_fn(self, batch):
		model_inputs = self.tokenizer([i for i in batch], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
		return {"model_inputs": model_inputs}


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', type=str, default='envclaim-climatebert')
	parser.add_argument('--outfile_name', type=str, default='environmental_claim_predictions.csv')
	args = parser.parse_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"

	# load model and tokenizer
	try:
		tokenizer = AutoTokenizer.from_pretrained(args.model_name)
	except:
		tokenizer = AutoTokenizer.from_pretrained("roberta-base")

	model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)

	# load sample sentences to predict

	with open("data/test.jsonl") as f:
		sentences = [json.loads(i)["text"] for i in f]

	predict_dataset = SequenceClassificationDataset(sentences, tokenizer)

	outputs = []
	probs = []
	with torch.no_grad():
		model.eval()
		for batch in tqdm(DataLoader(predict_dataset, batch_size=32, collate_fn=predict_dataset.collate_fn)):
			output = model(**batch["model_inputs"])
			logits = output.logits
			outputs.extend(logits.argmax(dim=1).tolist())
			probs.extend(logits.softmax(dim=1)[:,1].tolist())
	
	# save to outfile
	import pandas as pd

	df = pd.DataFrame(list(zip(sentences, outputs, probs)), columns=["sentence", "classification", "probability"])
	df.to_csv(args.outfile_name, index=False)

