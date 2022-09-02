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

## Pre-trained models

ClimateBERT fine-tuned on environmental claims (pytorch model) can be downloaded here:
* [envclaim_climatebert](https://www.dropbox.com/s/nww9lyihnyh7119/envclaim_climatebert.zip?dl=0)

We plan to host the model and dataset via huggingface transformers hub in the near future!

## Inference

run inference script with sample data

```shell
python src/inference_script.py
```

overwrite section "load sample sentences to predict" in inference_script.py for predicting customized data


## Replicate main experiments

### baseline experiments (majority, random and tf-idf)

```shell
python src/baselines.py 
```

### transformer models

```shell
pytohn src/transformer_models.py --do_save --save_path distilroberta-envclaim
```

(this also saves the resulting fine-tuned model in directory --save_path)

## Questions

If anything should not work or is unclear, please don't hesitate to contact the authors

* Dominik Stammbach (dominsta@ethz.ch)

