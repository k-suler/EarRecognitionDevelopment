# Ear Recognition model
Model uses VGG 19 deep neural network https://keras.io/api/applications/vgg/


Model can be downloaded from
https://www.mediafire.com/file/zyukaudnnvuxse7/finetuned_model_100e4.pt/file

put it in `./models/` folder.

## Results

CMC of the model:

![CMC](./results/Figure_1.png)

## Install

```
pip install -r requirenments.txt
```

## Train

To train new model run

```
python train.py
```

## Evaluation

```
python main.py
```
