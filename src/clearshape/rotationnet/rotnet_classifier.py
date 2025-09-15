import os
import shutil
import time

import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchmetrics.classification import Accuracy
from torchmetrics import F1Score

import numpy as np
import pandas as pd
from pathlib import Path

from clearshape.rotationnet.cnn import FineTuneModel
import clearshape.constants as constants


class RotationNetModel(pl.LightningModule):

    def __init__(self, arch='alexnet', num_classes=40, pretrained=False, criterion=None):
        super(RotationNetModel, self).__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.nview = 20
        self.vcand = np.load(constants.PATHS.ROOT / 'src/clearshape/rotationnet/vcand_case2.npy')

        if self.pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        self.model = FineTuneModel(model, arch, (num_classes + 1) * self.nview)
        # self.model.features = torch.nn.DataParallel(self.model.features)
        # self.model.cuda()

        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = F1Score(task="multiclass", num_classes=num_classes)


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # every sample contains list with 20 pictures, input needs to be without the list around and targets needs to be expanded for each picture
        x, y, _ = batch
        B, V, C, H, W = x.shape
        x = x.view(B*V, C, H, W)
        input = x#.view(-1, 256 * 13 * 13)
        target = y.repeat_interleave(20)

        nsamp = B#int( input.size(0) / self.nview )

        # input_var = torch.autograd.Variable(input)
        target_ = torch.LongTensor( target.size(0) * self.nview )

        # compute output
        output = self.model(input)
        num_classes = int( output.size( 1 ) / self.nview ) - 1
        output = output.view( -1, num_classes + 1 )

        ###########################################
        # compute scores and decide target labels #
        ###########################################
        output_ = torch.nn.functional.log_softmax( output )
        # divide object scores by the scores for "incorrect view label" (see Eq.(5))
        output_ = output_[ :, :-1 ] - torch.t( output_[ :, -1 ].repeat( 1, output_.size(1)-1 ).view( output_.size(1)-1, -1 ) )
        # reshape output matrix
        output_ = output_.view( -1, self.nview * self.nview, num_classes )
        output_ = output_.data.cpu().numpy()
        output_ = output_.transpose( 1, 2, 0 )
        # initialize target labels with "incorrect view label"
        for j in range(target_.size(0)):
            target_[ j ] = num_classes
        # compute scores for all the candidate poses (see Eq.(5))
        scores = np.zeros( ( self.vcand.shape[ 0 ], num_classes, nsamp ) )
        for j in range(self.vcand.shape[0]):
            for k in range(self.vcand.shape[1]):
                scores[ j ] = scores[ j ] + output_[ self.vcand[ j ][ k ] * self.nview + k ]
        # for each sample #n, determine the best pose that maximizes the score for the target class (see Eq.(2))
        for n in range( nsamp ):
            j_max = np.argmax( scores[ :, target[ n * self.nview ], n ] )
            # assign target labels
            for k in range(self.vcand.shape[1]):
                target_[ n * self.nview * self.nview + self.vcand[ j_max ][ k ] * self.nview + k ] = target[ n * self.nview ]
        ###########################################

        target_ = target_.cuda()
        target_var = torch.autograd.Variable(target_)

        # compute loss
        loss = self.criterion(output, target_var)

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
 
        B, V, C, H, W = x.shape 

        x = x.view(B*V, C, H, W)
        input = x#.view(-1, 256 * 13 * 13)
        target = y.repeat_interleave(V)

        target = target.cuda()
        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = self.model(input)
        val_loss = self.criterion(output, target)

        # log_softmax and reshape output
        num_classes = int( output.size( 1 ) / self.nview ) - 1
        output = output.view( -1, num_classes + 1 )
        output = torch.nn.functional.log_softmax( output )
        output = output[ :, :-1 ] - torch.t( output[ :, -1 ].repeat( 1, output.size(1)-1 ).view( output.size(1)-1, -1 ) )
        output = output.view( -1, self.nview * self.nview, num_classes )

        # measure accuracy and record loss
        preds = self.my_pred(output.data, target)

        self.val_acc(preds, y)
        f1_score = self.f1_score(preds, y)

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_f1_score', f1_score, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])

    # Test Step anpassen
    def test_step(self, batch, batch_idx):
        x, y = batch
        input = x.view(-1, 256 * 13 * 13)
        target = y.repeat_interleave(20)

        target = target.cuda()
        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = self.model(input)
        test_loss = self.criterion(output, target)

        # log_softmax and reshape output
        num_classes = int( output.size( 1 ) / self.nview ) - 1
        output = output.view( -1, num_classes + 1 )
        output = torch.nn.functional.log_softmax( output )
        output = output[ :, :-1 ] - torch.t( output[ :, -1 ].repeat( 1, output.size(1)-1 ).view( output.size(1)-1, -1 ) )
        output = output.view( -1, self.nview * self.nview, num_classes )

        # measure accuracy and record loss
        preds = self.my_pred(output.data, target)

        # Metriken berechnen
        self.log("test_loss", test_loss)
        self.log("test_acc", self.val_acc(preds, y))
        return test_loss

    # Predict Step anpassen
    def predict_step(self, batch):
        x = batch
        B, V, C, H, W = x.shape
        x = x.view(B*V, C, H, W)
        input = x#.view(-1, 256 * 13 * 13)
        # Create a dummy y vector with shape (B,)
        y = torch.zeros(B, dtype=torch.long, device=x.device)
        target = y.repeat_interleave(20)

        target = target.cuda()
        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = self.model(input)
        test_loss = self.criterion(output, target)

        # log_softmax and reshape output
        num_classes = int( output.size( 1 ) / self.nview ) - 1
        output = output.view( -1, num_classes + 1 )
        output = torch.nn.functional.log_softmax( output )
        output = output[ :, :-1 ] - torch.t( output[ :, -1 ].repeat( 1, output.size(1)-1 ).view( output.size(1)-1, -1 ) )
        output = output.view( -1, self.nview * self.nview, num_classes )

        # measure accuracy and record loss
        preds = self.my_pred(output.data, target)
        return preds

    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=constants.ROTNET.lr,
                                    momentum=constants.ROTNET.momentum,
                                    weight_decay=constants.ROTNET.weight_decay)
        return optimizer
    
    def my_pred(self, output_, target, topk = (1,)):
        maxk = max(topk)
        target = target[0:-1:self.nview]
        batch_size = target.size(0)

        num_classes = output_.size(2)
        output_ = output_.cpu().numpy()
        output_ = output_.transpose( 1, 2, 0 )
        scores = np.zeros( ( self.vcand.shape[ 0 ], num_classes, batch_size ) )
        output = torch.zeros( ( batch_size, num_classes ) )
        # compute scores for all the candidate poses (see Eq.(6))
        for j in range(self.vcand.shape[0]):
            for k in range(self.vcand.shape[1]):
                scores[ j ] = scores[ j ] + output_[ self.vcand[ j ][ k ] * self.nview + k ]
        # for each sample #n, determine the best pose that maximizes the score (for the top class)
        for n in range( batch_size ):
            j_max = int( np.argmax( scores[ :, :, n ] ) / scores.shape[ 1 ] )
            output[ n ] = torch.FloatTensor( scores[ j_max, :, n ] )
        output = output.cuda()

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()[0]
    
        return pred

    