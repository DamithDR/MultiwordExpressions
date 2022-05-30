
# MultiWord Expression Detection : MWE detection using transformers based models
Transformers based approach for multiword expressions detection.

## Installation
Then you need to install PyTorch. The recommended PyTorch version is 1.11.0
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) for more details specifically for the platforms.

When PyTorch has been installed, you can install requirements from source by cloning the repository and running:

```bash
git clone https://github.com/DamithDR/MultiwordExpressions.git
cd MultiwordExpressions
pip install -r requirements.txt
```

## Experiment Results
You can easily run experiments using following command and altering the parameters as you wish

```bash
python -m examples.mwe.en.run --model xlm-roberta-base
```

## Parameters
Please find the detailed descriptions of the parameters
```text
--model               : The name of the model you want to experiment with
```

## Model Names supported
```text
xlm-roberta-base
xlnet-base-cased
roberta-base
bert-base-multilingual-cased
bert-base-multilingual-uncased
bert-base-uncased
bert-base-cased
google/electra-base-discriminator
```

