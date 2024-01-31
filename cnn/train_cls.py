""" Classification experiment script

This script was designed to run on supercomputers/clusters.

v6: 2023-06-06 jakelee - refactor/cleanup, ported to cmutils
v7: 2024-01-30 jakelee - cleanup for delivery

Jake Lee, jakelee, jake.h.lee@jpl.nasa.gov
"""
import os.path as op
import csv
import argparse
import logging
import random
from datetime       import datetime
from tqdm           import tqdm
from pathlib import Path

from sklearn.metrics        import precision_recall_curve, classification_report, precision_recall_fscore_support
import numpy                as np
import wandb

import torch
from torch.nn       import CrossEntropyLoss
from torch.optim    import SGD, AdamW
from torchvision    import transforms

from archs.googlenetAA import googlenetAA

import cmutils
import cmutils.pytorch as cmtorch

from sam.sam import SAM
from sam.example.utility.bypass_bn import disable_running_stats, enable_running_stats
from sam.example.utility.step_lr import StepLR

# constant random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def get_augment(aug):
    """Define dataset preprocessing and augmentation"""

    preproc = transforms.Compose([
        cmtorch.ClampScaleMethaneTile()
    ])

    if aug:
        augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])
    else:
        augment = None

    return preproc, augment

def build_dataloader(csv_path, root='/', shuffle=True, augment=True, batch_size=16, num_workers=4):
    """ Build a pytorch dataloader based on provided arguments
    
    These preprocessing steps include calculated mean and standard deviation
    values for the _training set_ of each tiled dataset.

    csv_path: str
        Path to csv path to be loaded for dataloader
    root: str
        Root path to be prepended to paths inside csv
    norm: str
        Key to define normalization statistics
    shuffle: bool
        Whether to shuffle the dataset. Defaults to True.
    usergb: bool
        Whether to use the RGB layers. Defaults to False.
    """

    # Load data CSV
    datarows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            datarows.append(row)
    
    # Calculate loss weights to deal with imbalanced dataset
    all_labels = [1 if int(r[1]) == 1 else 0 for r in datarows]
    loss_weights = [1, (len(all_labels) - sum(all_labels)) / sum(all_labels)]

    # Define transforms and dataset class
    dataset = cmtorch.SegmentClassifyDatasetCH4(
        root,
        datarows,
        *get_augment(augment)
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader, loss_weights

def predict_loss(model,inputs,targets,lossfn):
    # GoogLeNet has two auxiliary branches for gradient stability
    if isinstance(model,GoogLeNet):
        pred, aux2, aux1 = model(inputs)

        p_loss = lossfn(pred, targets)
        x1_loss = lossfn(aux2, targets)
        x2_loss = lossfn(aux1, targets)

        # TODO: Using default mix values
        loss = p_loss + 0.3 * x1_loss + 0.3 * x2_loss
    else:
        pred = model(inputs)
        loss = lossfn(pred, targets)
        
    return pred, loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model on tiled methane data.")

    parser.add_argument('traincsv',         help="Filepath of the training set CSV")
    parser.add_argument('valcsv',           help="Filepath of the validation set CSV")
    parser.add_argument('--project',        help="Project name for wandb",
                                            type=str,
                                            default='Default')
    parser.add_argument('--exp',            help="Run name for wandb",
                                            type=str,
                                            default=None)
    parser.add_argument('--dataroot',       help="Root directory for relative paths. Defaults to / for absolute paths.",
                                            type=str,
                                            default='/')
    parser.add_argument('--model',          type=str,
                                            help="Model architecture to train",
                                            default="googlenetAA",
                                            choices=['googlenetAA'])
    parser.add_argument('--lr',             type=float,
                                            help="Learning rate",
                                            default=0.0001)
    parser.add_argument('--epochs',         type=int,
                                            default=100,
                                            help="Epochs for training")
    parser.add_argument('--batch',          type=int,
                                            default=16,
                                            help="Batch size for model training")
    parser.add_argument('--outroot',        default="train_out/",
                                            help="Root of output directories")
    parser.add_argument('--no-sam',         action='store_true',
                                            help="Disable SAM")
    parser.add_argument('--gpu',            type=int,
                                            default=0,
                                            help="GPU index to use")
    parser.add_argument('--wandb-dir',      help='Output directory for wandb logs. Defaults to ./wandb',
                                            type=str,
                                            default='./wandb')

    args = parser.parse_args()

    # SETUP ####################################################################

    # Set up output directories and files
    traincsv_parts = Path(args.traincsv).parts
    if len(traincsv_parts) >= 2:
        trainid = traincsv_parts[-2] + '_' + Path(traincsv_parts[-1]).stem
    else:
        trainid = traincsv_parts[-1].stem
    if not args.exp:
        expname = f"cls_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        expname = args.exp
    
    cmutils.check_mkdir(args.outroot)
    outdir = op.join(args.outroot, expname)
    cmutils.check_mkdir(outdir)
    cmutils.check_mkdir(op.join(outdir, 'weights'))

    # Training progress log and WandB
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(op.join(outdir, 'out.log')),
            logging.StreamHandler()
        ]
    )

    run = wandb.init(
        project = args.project,
        name = expname,
        dir = args.wandb_dir,
        config = {
            'expname': expname,
            'learning_rate': args.lr,
            'max_epochs': args.epochs,
            'batch_size': args.batch,
            'gpu': args.gpu,
            'architecture': args.model
        }
    )

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # apple silicon
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # DATA #####################################################################

    # Get dataloaders and loss weights
    train_loader, loss_weights = build_dataloader(args.traincsv,
                                                root=args.dataroot,
                                                shuffle=True,
                                                augment=True,
                                                batch_size=args.batch)

    val_loader, _ = build_dataloader(args.valcsv,
                                    root=args.dataroot,
                                    shuffle=False,
                                    augment=False,
                                    batch_size=args.batch)


    # MODEL ####################################################################

    #num_channels=4 if args.use_rgb else 1
    num_channels=1
    
    ## Load Model
    if args.model == 'googlenetAA':
        model = googlenetAA(pretrained=False,
                            num_channels=num_channels,
                            num_classes=2,
                            aux_logits=True,
                            init_weights=True)

    model.to(device)
    
    ## Set up optimizer
    if not args.no_sam:
        logging.info("Training with SAM")
        optimizer = SAM(model.parameters(),
                        SGD,
                        rho=2.0,
                        adaptive=True,
                        lr=args.lr,
                        momentum=0.9,
                        weight_decay=0.0005)
        scheduler = StepLR(optimizer, args.lr, args.epochs)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = None

    ## Loss Function
    logging.info(f"Using class weights {loss_weights}")
    ce_loss = CrossEntropyLoss(weight=torch.as_tensor(loss_weights,device=device))

    # make output dirs last to avoid polluting outdir with empty directories
    cmutils.check_mkdir(op.join(outdir))
    cmutils.check_mkdir(op.join(outdir, 'weights'))


    # TRAIN ####################################################################
    for epoch in range(args.epochs):
        model.train()

        epoch_loss = 0
        for iter, batch in enumerate(train_loader):
            inputs = batch['x'].to(device)
            targets = batch['class'].to(device)

            if not args.no_sam:
                # SAM training gets two steps per iteration
                # first forward-backward step
                enable_running_stats(model)

                pred, loss = predict_loss(model,inputs,targets,ce_loss)

                loss.mean().backward()

                optimizer.first_step(zero_grad=True)

                # second forward-backward step
                disable_running_stats(model)
                pred, loss = predict_loss(model,inputs,targets,ce_loss)

                loss.mean().backward()

                optimizer.second_step(zero_grad=True)
            else:
                # Standard training without SAM
                optimizer.zero_grad()
                pred, loss = predict_loss(model,inputs,targets,ce_loss)

                loss.mean().backward()
                
                optimizer.step()

            with torch.no_grad():
                # Keeping track of losses
                loss = loss.cpu().item()
                epoch_loss += loss
                run.log({
                    "loss_train": loss
                })
        
        if scheduler is not None:
            scheduler(epoch)

        # End of training for epoch
        epoch_loss = epoch_loss / len(train_loader)

        # Validation at each epoch
        model.eval()
        val_epoch_loss = 0
        val_epoch_true = []
        val_epoch_pred = []
        for iter, batch in enumerate(val_loader):
            inputs = batch['x'].to(device)
            targets = batch['class'].to(device)

            with torch.no_grad():
                # Note that using eval model disables aux returns
                pred = model(inputs)
                loss = ce_loss(pred, targets).cpu().item()
                prob = torch.nn.functional.softmax(pred, dim=1)
                val_epoch_true += targets.cpu().tolist()
                val_epoch_pred += [x[1] for x in prob.cpu().tolist()]

                val_epoch_loss += loss

        # End of validation for epoch
        val_epoch_loss = val_epoch_loss / len(val_loader)
        val_epoch_true = np.array(val_epoch_true)
        val_epoch_pred = np.array(val_epoch_pred)
        prec_50, recall_50, f1_50, _ = precision_recall_fscore_support(val_epoch_true, val_epoch_pred > 0.5, pos_label=1, average='binary')

        val_prec, val_recall, val_thresh = precision_recall_curve(val_epoch_true, val_epoch_pred)
        val_f1 = [2 * (p * r) / (p + r) for p, r in zip(val_prec, val_recall)]
        # Determine threshold for max f1
        best_thresh = val_thresh[np.argmax(val_f1)]
        prec_best, recall_best, f1_best, _ = precision_recall_fscore_support(val_epoch_true, val_epoch_pred > best_thresh, pos_label=1, average='binary')


        run.log({
            "val_loss_mean": val_epoch_loss,
            "val_precision_50": prec_50,
            "val_recall_50": recall_50,
            "val_f1_50": f1_50,
            "val_precision_best": prec_best,
            "val_recall_best": recall_best,
            "val_f1_best": f1_best,
            "train_loss_mean": epoch_loss,
            "epoch": epoch
        })

        # Save weights every 5 epochs
        if (epoch + 1) % 5 == 0:
            weightpath = op.join(outdir, 'weights', f"{epoch}_{expname}_weights.pt")
            torch.save(model.state_dict(), weightpath)

    wandb.finish()