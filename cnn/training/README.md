# CNN Model Training

## Description

This is a simplified version of the trainings script used to train the plume
detection models.

## Dependencies

* `tqdm`
* `sklearn`, `pytorch`, `torchvision`
* `numpy`, `matplotlib`, `PIL`
* `rasterio`, `GDAL`

## References

* [GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)
* [Sharpness Aware Minimization (SAM)](https://arxiv.org/pdf/2010.01412.pdf)

## Expected Dataset Structure

```
.
├── ang20200708t192518_ch4mf_v2y1_img
│   ├── bg    // background tiles (0)
│   ├── neg   // false positive tiles (-1)
│   └── pos   // true positive tiles (1)
│       ├── *.tif           // these are ENVI IMG pretending to be GTiff
│       ├── *.tif.aux.xml
│       └── *.hdr
├── train.csv         // tilepath relative to ./ , label [-1,0,1] for training set
├── test.csv          // same for test set
└── data_labels.csv   // same for entire dataset
```

## Usage

```
$ python experiment_script_all.py -h
usage: experiment_script_all.py [-h] [--lr LR] [--augment AUGMENT]
                                [--crop CROP] [--epochs EPOCHS]
                                [--outroot OUTROOT] [--no-sam] [--gpu GPU]
                                [--train-all]
                                dataroot {CalCH4_v8,COVID_QC,Permian_QC}

Train a classification model on tiled methane data.

positional arguments:
  dataroot              Directory path to dataset root
  {CalCH4_v8,COVID_QC,Permian_QC}
                        Campaign to train & test on

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Learning rate
  --augment AUGMENT     Data augmentation option
  --crop CROP           Center-crop the input tiles
  --epochs EPOCHS       Epochs for training
  --outroot OUTROOT     Root of output directories
  --no-sam              Disable SAM
  --gpu GPU             Specify GPU index to use
  --train-all           Train on the entire dataset
```

## Products

* `batch_losses.csv`
  * Batch-level training losses during training
* `epoch_losses.csv`
  * Epoch-level training losses during training
* `val_losses.csv`
  * Epoch-level validation losses during training
* `loss_curve.png`
  * Loss curve during training
* `train_predictions.csv`
  * Training set predictions with the final model based on best F1 threshold
* `val_predictions.csv`
  * Validation set predictions with the final model based on above threshold
* `train_report.txt`
  * Classification performance report on the training set
* `val_report.txt`
  * Classifiation performance report on the validation set
* `weights/`
  * Model weights at 5 epoch intervals

## Examples

### COVID_QC model
```
$ python experiment_script_all.py \
/path/to/dataset/root COVID_QC \
--lr 0.0001 \
--augment augB \
--crop 256 \
--epochs 100 \
--outroot /path/to/output/ \
--gpu 0

Using SAM
Using class weights [1, 22.64804469273743]
epoch 0, batch 0/265, train loss 0.6138716340065002
epoch 0, batch 1/265, train loss 0.6069806218147278
epoch 0, batch 2/265, train loss 0.5967214107513428
epoch 0, batch 3/265, train loss 0.7569172978401184
epoch 0, batch 4/265, train loss 0.5965495109558105
epoch 0, batch 5/265, train loss 0.9102346301078796
...
```

### CalCH4_v8 model
```
$ python experiment_script_all.py \
/path/to/dataset/root CalCH4_v8 \
--lr 0.0001 \
--augment augB \
--crop 256 \
--epochs 100 \
--outroot /path/to/output/ \
--gpu 0

Using SAM
Using class weights [1, 15.121212121212121]
epoch 0, batch 0/233, train loss 0.5584829449653625
epoch 0, batch 1/233, train loss 0.5317976474761963
epoch 0, batch 2/233, train loss 0.7786059975624084
epoch 0, batch 3/233, train loss 0.8385669589042664
epoch 0, batch 4/233, train loss 0.7517789006233215
epoch 0, batch 5/233, train loss 0.7652335166931152
...
```

### Permian_QC model
```
Using SAM
Using class weights [1, 9.429236499068901]
epoch 0, batch 0/701, train loss 0.6808735132217407
epoch 0, batch 1/701, train loss 0.7842758297920227
epoch 0, batch 2/701, train loss 0.8541020154953003
epoch 0, batch 3/701, train loss 0.7854573726654053
epoch 0, batch 4/701, train loss 0.634735107421875
epoch 0, batch 5/701, train loss 0.6570754051208496
```
