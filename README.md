# environmental_claims

This repo contains the code, dataset and models for our working paper paper [(Stammbach et al., 2022)](https://arxiv.org/abs/2209.00507)

## Installation

Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/) and linux, the environment can be installed with the following command:
```shell
conda create -n environmental_claims python=3.6
conda activate environmental_claims

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

## Pre-trained models and Data

* Our dataset is stored the folder "data". 
* A fine-tuned ClimateBERT pytorch model on our dataset can be downloaded here: [climatebert-environmental-claims](https://www.dropbox.com/s/opyj49dw36tkmko/climatebert-environmental-claims.zip?dl=0)

We also host the dataset and model on huggingface
* [huggingface dataset](https://huggingface.co/datasets/climatebert/environmental_claims)
* [huggingface model](https://huggingface.co/climatebert/environmental-claims)

In our paper and dataset, we discard sentences where 2 annotators say a sentence "is an environmental claim", but 2 annotators disagree and therefore we have a tie.
We host the [full dataset here](https://www.dropbox.com/s/gbmb9p4epifbpv9/all_3000_environmental_claims.json?dl=0), including all 3000 sentences, and agreement between annotators (either 0.5, 0.75 or 1.0). Labels are:
* "yes" for environmental claims
* "no" for others
* "tie" if a datapoints has an agreement of 0.5

 
## Inference

To predict environmental claims in custom data, we provide an inference script. For running the script with some data (either a "jsonl" file with a column "sentences" or "text", or a ".txt" file with one sentence by line"), simply run the following python command.

```shell
python src/inference_script.py --filename data/test.jsonl --model_name climatebert-environmental-claims --outfile_name environmental_claim_predictions.csv
```

## Replicate main experiments in our paper

### baseline experiments (majority, random and tf-idf)
To replicate the baseline experiments, run the following python script.

```shell
python src/baselines.py 
```

(this prints the rows in our Table 2 for these experiments)

### transformer models

To fine-tune a climatebert model on our dataset, run the following python script.
```shell
python transformer_models.py --do_save --save_path climatebert-environmental-claims --model_name climatebert/distilroberta-base-climate-f
```

(this also saves the resulting fine-tuned model in directory --save_path)

## Questions

If anything should not work or is unclear, please don't hesitate to contact the authors

* Dominik Stammbach (dominsta@ethz.ch)

