""" Segmentation experiment script

This script was designed to run on supercomputers/clusters.

v1: 2023-06-06 jakelee - fresh rewrite with cmutils
v2: 2023-09-20 jakelee - wandb integration, multicampaign
v3: 2024-01-24 jakelee - loss function sweep, U-Net, UPerNet comparison
v4: 2024-01-30 jakelee - Cleanup for delivery

Jake Lee, jakelee, jake.h.lee@jpl.nasa.gov
"""
import os.path as op
import csv
import argparse
import logging
import random
from datetime       import datetime
from pathlib import Path
from collections import OrderedDict

from sklearn.metrics        import precision_recall_curve
from sklearn.metrics        import precision_recall_fscore_support as prfs

import numpy                as np
import wandb

import torch
from torch import nn
from torch.functional import F
from torch.nn       import BCEWithLogitsLoss
from torchvision    import transforms
from torch.optim    import AdamW

from archs.unet import DeepPaddedUNet
from archs.upernet import UperNet
import cmutils
import cmutils.pytorch as cmtorch

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

def build_dataloader(csv_path, root='/', shuffle=True, augment=True, batch_size=8):
    """ Build a pytorch dataloader based on provided arguments
    
    These preprocessing steps include calculated mean and standard deviation
    values for the _training set_ of each tiled dataset.

    csv_path: str
        Path to csv path to be loaded for dataloader
    root: str
        Root path to be prepended to paths inside csv
    shuffle: bool
        Whether to shuffle the dataset. Defaults to True.
    augment: bool
        Whether to add flip augmentation. Defaults to True.
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

    dataset = cmtorch.SegmentClassifyDatasetCH4(
        root,
        datarows,
        *get_augment(augment)
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4
    )

    return dataloader, loss_weights

def tilewise_accuracy_new(lab_masks,seg_probs,prob_thr=None,cls_agg='max',
                      verbose=False):
    """
    Summary: given an array of label images and matching array of
    pixelwise probability scores:
    - convert pixelwise to tilewise labels + predictions
    - evalutate prfs on resulting tilewise labels + predictions
    Both input arrays are 4-d arrays with matching dimensions = 
    [n_imgs,1,n_rows,n_cols]
    Arguments:
    - lab_masks: uint8 array of binary pixelwise labels \in {0,1}
    - seg_probs: float32 array of pixelwise probability scores \in [0,1]
    - prob_thr: minimum pixelwise probability for positive class membership
    - cls_agg: method to convert pixelwise class probabilities to tilewise class probability (max,median,mean)
    Output:
    - cls_out: tilewise classification metrics + metadata
    - seg_out: pixelwise segmentation metrics + metadata
    """

    if lab_masks.ndim==3:
        lab_masks = lab_masks[:,np.newaxis]
    if seg_probs.ndim==3:
        seg_probs = seg_probs[:,np.newaxis]
    
    assert lab_masks.shape == seg_probs.shape, f'lab_masks + seg_probs shape mismatch ({lab_masks.shape} vs. {seg_probs.shape})'
    assert lab_masks.shape[1] == 1, f'multi channel images of dim={lab_masks.shape[1]} not supported currently'

    n_img = lab_masks.shape[0]
    cls_labs = np.empty(n_img,dtype=np.uint8)
    cls_prob = np.empty(n_img,dtype=np.float32)


    lab_masks = lab_masks==1
    seg_labs = lab_masks.ravel()
    seg_prob = seg_probs.ravel()

    cls_labs = lab_masks.any(axis=(2,3))
    cls_prob = seg_probs.max(axis=(2,3))

        
    # classification metrics 
    pre_,rec_,thr_ = precision_recall_curve(cls_labs,cls_prob,
                                            pos_label=1)
    cls_thr = prob_thr
    if cls_thr is None:
        f1b_ = pre_+rec_
        f1b_ = 2*(pre_*rec_)/np.where(f1b_!=0,f1b_,1)
        cls_thr = thr_[np.argmax(f1b_)]

    # cls apply threshold
    cls_pred = cls_prob>=cls_thr
    # cls count err
    cls_errs = cls_pred != cls_labs

    # cls n tp, fp, fn, tn
    cls_ntp = (~cls_errs &  cls_labs).sum()
    cls_nfp = ( cls_errs & ~cls_labs).sum()
    cls_nfn = ( cls_errs &  cls_labs).sum()
    cls_ntn = (~cls_errs & ~cls_labs).sum()

    # cls av. precision
    cls_ap = -np.sum(pre_[:-1]*np.diff(rec_))
    
    # cls precision recall f1 from binary
    cls_prf = prfs(cls_labs,cls_pred,pos_label=1,average='binary',
                   zero_division=0)[:-1]

    # to calculate median positive and negative probabilities
    cls_pos = cls_prob[cls_labs]
    cls_neg = cls_prob[~cls_labs]
    cls_pos = cls_pos[np.isfinite(cls_pos)]
    cls_neg = cls_neg[np.isfinite(cls_neg)]

    cls_out = dict(ap=cls_ap,
                   prob_thr=cls_thr,
                   pre=cls_prf[0],
                   rec=cls_prf[1],
                   f1b=cls_prf[2],
                   ntp=cls_ntp,
                   nfp=cls_nfp,
                   nfn=cls_nfn,
                   ntn=cls_ntn,
                   npos=cls_ntp+cls_nfn,
                   nneg=cls_ntn+cls_nfp,
                   prob_pos=np.nanmedian(cls_pos),
                   prob_neg=np.nanmedian(cls_neg),
    )

    # segmentation metrics
    pre_,rec_,thr_ = precision_recall_curve(seg_labs,seg_prob,
                                            pos_label=1)

    # modified f1 calculation because f1 can be nan sometimes
    f1b_ = pre_+rec_
    f1b_ = 2*(pre_*rec_)/np.where(f1b_!=0,f1b_,1)
    f1b_max = np.nanmax(f1b_)
    f1b_thr = np.unique(thr_[f1b_[:-1]==f1b_max])
    seg_thr = f1b_thr[-1]

    seg_pred = seg_prob>=seg_thr
    seg_errs = seg_labs!=seg_pred
    seg_ntp = (~seg_errs &  seg_labs).sum()
    seg_nfp = ( seg_errs & ~seg_labs).sum()
    seg_nfn = ( seg_errs &  seg_labs).sum()
    seg_ntn = (~seg_errs & ~seg_labs).sum()

    # seg av. precision
    seg_ap = -np.sum(pre_[:-1]*np.diff(rec_))
    # seg precision recall f1 from binary
    seg_prf = prfs(seg_labs,seg_pred,pos_label=1,average='binary',
                        zero_division=0)[:-1]
    
    # to calculate median positive and negative probabilities
    seg_pos = seg_prob[seg_labs]
    seg_neg = seg_prob[~seg_labs]
    seg_pos = seg_pos[np.isfinite(seg_pos)]
    seg_neg = seg_neg[np.isfinite(seg_neg)]

    seg_out = dict(ap=seg_ap,
                   prob_thr=seg_thr,
                   pre=seg_prf[0],
                   rec=seg_prf[1],
                   f1b=seg_prf[2],
                   ntp=seg_ntp,
                   nfp=seg_nfp,
                   nfn=seg_nfn,
                   ntn=seg_ntn,
                   npos=seg_ntp+seg_nfn,
                   nneg=seg_ntn+seg_nfp,
                   prob_pos=np.nanmedian(seg_pos),
                   prob_neg=np.nanmedian(seg_neg),
    )
    
    return seg_out,cls_out


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha  = alpha
        self.gamma  = gamma
        self.bce_loss = BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(self, x, y):
        bce_loss = self.bce_loss(x,y)
        focal_loss = self.alpha * (1-torch.exp(-bce_loss))**self.gamma * bce_loss
        return focal_loss

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1, smooth=1):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        #Old version was flipped
        Tversky = (TP + self.smooth) / (TP + self.beta*FP + self.alpha*FN + self.smooth)  
        FocalTversky = (1 - Tversky)**self.gamma
                       
        return FocalTversky


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a segmentation model on tiled methane data.")

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
    parser.add_argument('--model',          help="Which model to train",
                                            choices=["DeepUNet", "UPerNet"])
    parser.add_argument('--lr',             type=float,
                                            help="Learning rate, U-Net default 0.001, UPerNet default 0.0001",
                                            default=None)
    parser.add_argument('--epochs',         type=int,
                                            default=200,
                                            help="Epochs for training. Default 200.")
    parser.add_argument('--batch',          type=int,
                                            default=16,
                                            help="Batch size for model training")
    parser.add_argument('--outroot',        default="train_out/",
                                            help="Root of output directories")
    parser.add_argument('--gpu',            type=int,
                                            default=0,
                                            help="Specify GPU index to use")
    parser.add_argument('--backbone',       help="Filepath to backbone weights. Defaults to models/multicampaign_googlenet.pt",
                                            type=str,
                                            default="models/multicampaign_googlenet.pt")
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
        expname = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        expname = args.exp

    cmutils.check_mkdir(args.outroot)
    outdir = op.join(args.outroot, expname)
    cmutils.check_mkdir(outdir)
    cmutils.check_mkdir(op.join(outdir, 'weights'))

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
            'batch_size': args.batch,
            'gpu': args.gpu,
            'seg_loss': args.seg_loss,
            'cls_loss': args.cls_loss,
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
    pos_cls = loss_weights[1]

    val_loader, _ = build_dataloader(args.valcsv,
                                    root=args.dataroot,
                                    shuffle=False,
                                    augment=False,
                                    batch_size=args.batch)


    # MODEL ####################################################################

    ## Load Model

    if not args.lr:
        if args.model == "DeepUNet":
            lr = 0.001
        elif args.model == "UPerNet":
            lr = 0.0001
    else:
        lr = args.lr

    if args.model == "DeepUNet":
        model = DeepPaddedUNet(in_ch=1, num_classes=1).to(device)
        optimizer = AdamW(model.parameters(), lr=lr)
    elif args.model == "UPerNet":
        model = UperNet(num_classes=1, in_channels=1, freeze_backbone=True).to(device)
        model.backbone.load_state_dict(
            torch.load(
                args.backbone,
                map_location=device),
            strict=False)
        optimizer = AdamW(filter(lambda p:p.requires_grad, model.get_decoder_params()), lr=lr)

    ## Loss Function
    seg_loss = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=4/3, smooth=1e-5)
    cls_loss = BCEWithLogitsLoss()

    # TRAIN ####################################################################
    iters = len(train_loader)
    for epoch in range(args.epochs):
        model.train()

        epoch_loss = 0
        epoch_loss_seg = 0
        epoch_loss_cls = 0
        for i, batch in enumerate(train_loader):
            inputs = batch['x'].to(device)
            targets = batch['y'].to(device)
            clstrgs = batch['class'].to(device).float()
            clstrgs = torch.unsqueeze(clstrgs, 1)

            # Standard training without SAM
            optimizer.zero_grad()

            # Model output
            segout = model(inputs)
            clsout = torch.flatten(torch.nn.AdaptiveMaxPool2d((1,1))(segout), 1)

            if cls_loss is None:
                clsl = torch.tensor(0)
            else:
                clsl = cls_loss(clsout, clstrgs)

            segl = seg_loss(segout, targets)
            loss = clsl + segl

            loss.mean().backward()

            optimizer.step()

            with torch.no_grad():
                epoch_loss += loss.cpu().item()
                epoch_loss_seg += segl.cpu().item()
                epoch_loss_cls += clsl.cpu().item()
            
            run.log({
                "loss_seg_train": segl.cpu().item(),
                "loss_cls_train": clsl.cpu().item(),
                "loss_train": loss.cpu().item()
            })

        #scheduler.step()
        epoch_loss = epoch_loss / len(train_loader)
        epoch_loss_seg = epoch_loss_seg / len(train_loader)
        epoch_loss_cls = epoch_loss_cls / len(train_loader)


        # Validation at each epoch
        # Disabling this running evaluation will reduce training time
        model.eval()
        val_labels = []
        val_prob = []
        val_epoch_loss = 0
        val_epoch_loss_cls = 0
        val_epoch_loss_seg = 0

        vis_ch4mf = []
        vis_label = []
        vis_seg = []
        vis_path = []

        for iter, batch in enumerate(val_loader):
            inputs = batch['x'].to(device)
            targets = batch['y'].to(device)
            clstrgs = batch['class'].to(device).float()
            clstrgs = torch.unsqueeze(clstrgs, 1)

            with torch.no_grad():
                segout = model(inputs)
                clsout = torch.flatten(torch.nn.AdaptiveMaxPool2d((1,1))(segout), 1)
                segprobs = torch.nn.functional.sigmoid(segout)

                #wandb
                val_labels += targets.cpu()
                val_prob += segprobs.cpu()
                for i in range(segprobs.size()[0]):
                    if len(vis_seg) < 4 and clstrgs[i] == 1:
                        vis_ch4mf.append(inputs[i])
                        vis_label.append(targets[i])
                        vis_seg.append(segprobs[i])
                        vis_path.append(Path(batch['xpath'][i]).name)

                if cls_loss is None:
                    clsl = torch.tensor(0)
                else:
                    clsl = cls_loss(clsout, clstrgs)
                segl = seg_loss(segout, targets)
                loss = clsl + segl

                val_epoch_loss += loss
                val_epoch_loss_cls += clsl.cpu().item()
                val_epoch_loss_seg += segl.cpu().item()

        # End of validation for epoch
        val_epoch_loss = val_epoch_loss.cpu() / len(val_loader)
        val_epoch_loss_cls = val_epoch_loss_cls / len(val_loader)
        val_epoch_loss_seg = val_epoch_loss_seg / len(val_loader)

        # Eval
        val_labels,val_prob = np.float32(val_labels),np.float32(val_prob)
        val_seg,val_cls = tilewise_accuracy_new(val_labels, val_prob,
                                            cls_agg='max',
                                            verbose=False)
        val_seg,val_cls = tilewise_accuracy_new(val_labels, val_prob,
                                            cls_agg='max',
                                            verbose=False)

        logdata = OrderedDict()
        logdata['epoch_loss/loss_train'] = epoch_loss
        logdata['epoch_loss/loss_train_seg'] = epoch_loss_seg
        logdata['epoch_loss/loss_train_cls'] = epoch_loss_cls
        logdata['epoch_loss/loss_val'] = val_epoch_loss
        logdata['epoch_loss/loss_val_seg'] = val_epoch_loss_seg
        logdata['epoch_loss/loss_val_cls'] = val_epoch_loss_cls
        
        log_metrics = ['ap','f1b','pre','rec']
        log_probs = ['prob_pos','prob_neg','prob_thr']

        for key in log_metrics:
            logdata['val_metrics/'+key+'_cls'] = val_cls[key]
            logdata['val_metrics/'+key+'_seg'] = val_seg[key]
        for key in log_probs:
            logdata['val_probs/'+key+'_cls'] = val_cls[key]
            logdata['val_probs/'+key+'_seg'] = val_seg[key]
        
        wandb.log(logdata)

        # Save weights every 10 epochs
        if (epoch + 1) % 10 == 0:
            weightpath = op.join(outdir, 'weights', f"{epoch}_{expname}_weights.pt")
            torch.save(model.state_dict(), weightpath)

    wandb.finish()
