# A TensorFlow Implementation of the Transformer: Attention Is All You Need

## Requirements
  * NumPy >= 1.14.0
  * TensorFlow >= 1.9

## Difference
[original](https://github.com/Kyubyong/transformer.git)  
This repository is converted tensorflow to tf.keras

## Training
* STEP 1. Download [IWSLT 2016 German–English parallel corpus](https://wit3.fbk.eu/download.php?release=2016-01&type=texts&slang=de&tlang=en) and extract it to `corpora/` folder.
```sh
wget -qO- --show-progress https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz | tar xz; mv de-en corpora
```
* STEP 2. Adjust hyper parameters in `hyperparams.py` if necessary.
* STEP 3. Run `prepro.py` to generate vocabulary files to the `preprocessed` folder.
* STEP 4. Run `train.py`

## Evaluation
  * Run `eval.py`.