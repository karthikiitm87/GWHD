from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import time
from tqdm.notebook import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt

# Torch imports 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.autograd import Function

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
from torchvision.ops.boxes import box_iou
from torchvision.models.detection._utils import Matcher
import torchvision.models as models
from torchvision.ops import nms, box_convert


# Albumentations is used for the Data Augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Pytorch import
from pytorch_lightning.core.module import LightningModule
#from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from collections import OrderedDict
from torchvision.models.detection.image_list import ImageList
from typing import List, Tuple


class WheatDataset(Dataset):
    """A dataset example for GWC 2021 competition."""

    def __init__(self, csv_file, root_dir, image_set, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional data augmentation to be applied
                on a sample.
        """

        annotations = pd.read_csv(csv_file)
        self.image_set = image_set
        self.image_path = root_dir+annotations["image_name"]
        self.boxes = [self.decodeString(item) for item in annotations["BoxesString"]]
        self.domains_str = annotations['domain']
        
        if(image_set == 'train'):
          self._domains = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17')
        elif(image_set == 'val'):
          self._domains = ('18', '19', '20', '21', '22', '23', '24', '25')
        else:
          self._domains = ('26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46')
        
        self.num_domains = len(self._domains)
        self._domain_to_ind = dict(zip(self._domains, range(len(self._domains))))
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        
        imgp = self.image_path[idx]
        bboxes = self.boxes[idx]
        img = cv2.imread(imgp)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Opencv open images in BGR mode by default
        
        try:
          domain = torch.tensor(self._domain_to_ind[str(self.domains_str[idx])])
        except:
          domain = torch.tensor(-1)
          
        try:
          if self.transform:
              transformed = self.transform(image=image,bboxes=bboxes,class_labels=["wheat_head"]*len(bboxes)) 
              image_tr = transformed["image"]/255.0
              bboxes = transformed["bboxes"]
        except:
          print(len(bboxes))
          print(imgp)
        if len(bboxes) > 0:
          bboxes = torch.stack([torch.tensor(item) for item in bboxes])
        else:
          bboxes = torch.zeros((0,4))
          
               
        return image_tr, bboxes, domain, image
              
    def decodeString(self,BoxesString):
      """
      Small method to decode the BoxesString
      """
      if BoxesString == "no_box":
          return np.zeros((0,4))
      else:
          try:
              boxes =  np.array([np.array([int(i) for i in box.split(" ")])
                              for box in BoxesString.split(";")])
              return boxes.astype(np.int32).clip(min=0)
          except:
              print(BoxesString)
              print("Submission is not well formatted. empty boxes will be returned")
              return np.zeros((0,4))


seed_everything(25081992)


#SIZE = 512
#This is for analysing the influence of augmentation on the performance
#All the individual augmentations are commented so as to get the true impact
#of baseline model. We can uncomment as per need. 
train_transform = A.Compose(
        [
        #A.HorizontalFlip(p=0.5),
        #A.VerticalFlip(p=0.5),
        #A.Transpose(p=0.5),
        #A.RandomRotate90(p=0.5),
        #A.RandomRotate90(A.RandomRotate90(p=1.0), p=0.5),
        #A.RandomRotate90(A.RandomRotate90(A.RandomRotate90(p=1.0), p=1.0), p=0.5),
        ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_area=20)
    )


valid_transform = A.Compose([
    ToTensorV2(p=1.0),
],p=1.0,bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_area=20))



def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

    """

    images = list()
    targets=list()
    orig_img = list()
    domain_labels = list()
    for i, t, d, io in batch:
        images.append(i)
        targets.append(t)
        orig_img.append(io)
        domain_labels.append(d)
    images = torch.stack(images, dim=0)

    return images, targets, domain_labels, orig_img




#import fasterrcnn
class myDAFasterRCNN(LightningModule):
    def __init__(self, n_classes, batchsize, n_vdomains):
        super(myDAFasterRCNN, self).__init__()
        self.n_classes = n_classes
        self.batchsize = batchsize
        self.n_vdomains = n_vdomains
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(n_classes = self.n_classes, min_size=1024, max_size=1024, weights_backbone=ResNet50_Weights.IMAGENET1K_V1)              
        #in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        #self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

        
	     
        self.best_val_acc = 0
        self.val_acc_stack = [[] for i in range(self.n_vdomains)]
        self.freq = torch.tensor(np.zeros(n_classes))
        self.log('val_loss', 100000)
        self.log('val_acc', self.best_val_acc)

        self.base_lr = 1e-5 #Original base lr is 1e-4
        self.momentum = 0.9
        self.weight_decay=0.0001
        
               
              
    def forward(self, imgs,targets=None):
      # Torchvision FasterRCNN returns the loss during training 
      # and the boxes during eval
      self.detector.eval()
      return self.detector(imgs)
    
    def configure_optimizers(self):
      
      optimizer = torch.optim.Adam([{'params': self.detector.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },],) 
      
      lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, threshold=0.0001, min_lr=0, eps=1e-08),
                      'monitor': 'val_loss'}
      
      
      return [optimizer], [lr_scheduler]
      
    def train_dataloader(self):
      num_train_sample_batches = len(tr_dataset)//self.batchsize 
      temp_indices = np.array([i for i in range(len(tr_dataset))])
      np.random.shuffle(temp_indices)
      sample_indices = []
      for i in range(num_train_sample_batches):
  
        batch = temp_indices[self.batchsize*i:self.batchsize*(i+1)]
  
        for index in batch:
          sample_indices.append(index)  
  
      return torch.utils.data.DataLoader(tr_dataset, batch_size=self.batchsize, sampler=sample_indices, shuffle=False, collate_fn=collate_fn, num_workers=4)
      
    def training_step(self, batch, batch_idx):
      
      imgs = list(image.cuda() for image in batch[0]) 

      targets = []
      for boxes, domain in zip(batch[1], batch[2]):
        target= {}
        target["boxes"] = boxes.float().cuda()
        target["labels"] = torch.ones(len(target["boxes"])).long().cuda()
        targets.append(target)

      # fasterrcnn takes both images and targets for training, returns
      #Detection using source images
      
      temp_loss = []
      for index in range(len(imgs)):
        detections = self.detector([imgs[index]], [targets[index]])
        temp_loss.append(sum(loss1 for loss1 in detections.values()))

               
      loss = torch.mean(torch.stack(temp_loss))
        
      return {"loss": loss}

    def validation_step(self, batch, batch_idx):
      img, boxes, domain, _ = batch
      
      preds = self.forward(img)
      preds[0]['boxes'] = preds[0]['boxes'][preds[0]['scores'] > 0.5]
      self.val_acc_stack[domain[0]].append(torch.stack([self.accuracy(b,pb["boxes"],iou_threshold=0.5) for b,pb in zip(boxes,preds)]))
    
    def validation_epoch_end(self, validation_step_outputs):

         
      temp = 0
      non_zero_domains = 0
      
      for item in range(len(self.val_acc_stack)):
        
        if(self.val_acc_stack[item]):
          temp = temp + torch.mean(torch.stack(self.val_acc_stack[item]))
          non_zero_domains = non_zero_domains + 1
          print(torch.mean(torch.stack(self.val_acc_stack[item])))
          
      temp = temp/non_zero_domains #8 Validation domains 
      self.log('val_loss', 1 - temp)  #Logging for model checkpoint
      self.log('val_acc', temp)
      if(self.best_val_acc < temp):
        self.best_val_acc = temp
        self.best_val_acc_epoch = self.trainer.current_epoch
      

      self.val_acc_stack = [[] for i in range(self.n_vdomains)]
      
      print('Validation ADA: ',temp)
      self.mode = 0

   
    def accuracy(self, src_boxes,pred_boxes ,  iou_threshold = 1.):
      """
      #The accuracy method is not the one used in the evaluator but very similar
      """
      total_gt = len(src_boxes)
      total_pred = len(pred_boxes)
      if total_gt > 0 and total_pred > 0:


        # Define the matcher and distance matrix based on iou
        matcher = Matcher(iou_threshold,iou_threshold,allow_low_quality_matches=False) 
        match_quality_matrix = box_iou(src_boxes,pred_boxes)

        results = matcher(match_quality_matrix)
        
        true_positive = torch.count_nonzero(results.unique() != -1)
        matched_elements = results[results > -1]
        
        #in Matcher, a pred element can be matched only twice 
        false_positive = torch.count_nonzero(results == -1) + ( len(matched_elements) - len(matched_elements.unique()))
        false_negative = total_gt - true_positive

            
        return  true_positive / (true_positive + false_positive + false_negative) 

      elif total_gt == 0:
          if total_pred > 0:
              return torch.tensor(0.).cuda()
          else:
              return torch.tensor(1.).cuda()
      elif total_gt > 0 and total_pred == 0:
          return torch.tensor(0.).cuda()
          
tr_dataset = WheatDataset('../datasets/Annots/official_train.csv', root_dir='../datasets/gwhd_2021/images/', image_set = 'train', transform=train_transform)
vl_dataset = WheatDataset('../datasets/Annots/official_val.csv', root_dir='../datasets/gwhd_2021/images/', image_set = 'val', transform=valid_transform)
train_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=2, shuffle=True,  collate_fn=collate_fn, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(vl_dataset, batch_size=1, shuffle=False,  collate_fn=collate_fn, num_workers=4)

           
import os
detector = myDAFasterRCNN(n_classes=2, batchsize=2, n_vdomains = 8)


NET_FOLDER = 'GWHD'
weights_file = 'best_baseline'
if(os.path.exists(NET_FOLDER+'/'+weights_file+'.ckpt')):
  detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])
else:	
  if not os.path.exists(NET_FOLDER):
    mode = 0o777
    os.mkdir(NET_FOLDER, mode)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
early_stop_callback= EarlyStopping(monitor='val_acc', min_delta=0.00, patience=10, verbose=False, mode='max')


checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=NET_FOLDER, filename=weights_file)
trainer = Trainer(accelerator='gpu', devices=1, max_epochs=100, deterministic=False, callbacks=[checkpoint_callback, early_stop_callback], reload_dataloaders_every_n_epochs=1)
trainer.fit(detector, train_dataloaders = train_dataloader, val_dataloaders=val_dataloader)


detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])
detector.freeze()
test_dataset = WheatDataset('../datasets/Annots/official_test.csv', root_dir='../datasets/gwhd_2021/images/', image_set = 'test', transform=valid_transform)

detector.detector.eval()
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
def acc_new(src_boxes, pred_boxes, iou_threshold = 1.):
      
      #The accuracy method is not the one used in the evaluator but very similar
      
      total_gt = len(src_boxes)
      total_pred = len(pred_boxes)
      if total_gt > 0 and total_pred > 0:


        # Define the matcher and distance matrix based on iou
        matcher = Matcher(iou_threshold,iou_threshold,allow_low_quality_matches=False) 
        match_quality_matrix = box_iou(src_boxes,pred_boxes)

        results = matcher(match_quality_matrix)
        
        true_positive = torch.count_nonzero(results.unique() != -1)
        matched_elements = results[results > -1]
        
        #in Matcher, a pred element can be matched only twice 
        false_positive = torch.count_nonzero(results == -1) + ( len(matched_elements) - len(matched_elements.unique()))
        false_negative = total_gt - true_positive

            
        return  true_positive / ( true_positive + false_positive + false_negative )

      elif total_gt == 0:
          if total_pred > 0:
              return torch.tensor(0.).cuda()
          else:
              return torch.tensor(1.).cuda()
      elif total_gt > 0 and total_pred == 0:
          return torch.tensor(0.).cuda()
          

detector.to('cuda')
val_acc_stack = [[] for i in range(test_dataset.num_domains)]
domain = torch.zeros(test_dataset.num_domains)
for index, data_sample in enumerate(iter(test_dataloader)):

  images, boxes, labels, orig_img = data_sample
   
  preds = detector(images.cuda())
  
  preds[0]['boxes'] = preds[0]["boxes"].detach().cpu()
  preds[0]['labels'] = preds[0]["labels"].detach().cpu()
  preds[0]['scores'] = preds[0]["scores"].detach().cpu()
  
  preds[0]['boxes'] = preds[0]['boxes'][preds[0]['scores'] > 0.5]
  val_acc_stack[labels[0]].append(torch.stack([acc_new(b,pb["boxes"],iou_threshold=0.5) for b,pb in zip(boxes,preds)]).detach().cpu())
      
  domain[labels[0]] = domain[labels[0]] + 1
  
  for box in preds[0]['boxes']:
    cv2.rectangle(orig_img[0], (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)
  
  for box in boxes[0]:
    cv2.rectangle(orig_img[0], (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
  
  # The predictions are saved with each domain. If you want your predictions to be saved then 
  # ensure you have appropriate predictions folder for the same.
  path = './GWHD/predictions_prop2/'+str(labels[0].item())+'/'+str(index)+'.png'
  cv2.imwrite(path, cv2.cvtColor(orig_img[0], cv2.COLOR_RGB2BGR))
  domain[labels[0]] = domain[labels[0]] + 1
  print(index)
  
      
#ADA_whole = torch.mean(torch.stack(val_acc_stack))

#print(ADA_whole)
weights = [1/domain[i] for i in range(test_dataset.num_domains)]
temp = 0
test_acc = []
for index in range(len(val_acc_stack)):

  if(len(val_acc_stack[index]) == 0):
    print(str(index)+'  is empty')
  else:
    temp = temp + weights[index]*torch.sum(torch.stack(val_acc_stack[index]))
    test_acc.append(torch.mean(torch.stack(val_acc_stack[index])).item())
    print(torch.mean(torch.stack(val_acc_stack[index])))
    
np.savetxt(NET_FOLDER+'/test_acc.txt',np.array(test_acc))
print('WAD:', torch.mean(torch.tensor(test_acc)))

