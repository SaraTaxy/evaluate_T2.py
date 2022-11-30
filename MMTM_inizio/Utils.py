import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import scipy.io as sio
from scipy.stats import zscore
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

import os
import shutil

from random import choice, sample, seed, randint, random, gauss
from scipy.io import loadmat
import skimage.morphology as skm
from torch.utils.data import ConcatDataset, DataLoader, Dataset
#from torchvision.datasets import DatasetFolder
import math

#rendere l'esecuzione deterministica 
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


x = loadmat('/content/DatasetDCE_Water_DWI_clinico/0_cavo/69_Bertucci_a_1072_1092_1200.mat')
lista_giusta = []
for element in x['nome_var'].tolist():
  lista_giusta.append(element.split(' ')[0])
print(lista_giusta)


lista_selezionata= ['eta', 'familiarita', 'HT', 'menopausa',
                    'dimensioni', 'ER','PgR', 'ki67','herb', 'G' ] 
IDX_TO_CONSIDER = []
for element in lista_selezionata:
  IDX_TO_CONSIDER.append(lista_giusta.index(element))

print(lista_selezionata)

#-------------------------------------------------------------------------------> Normalizzazione durante la lettura
def normalizeZscore(x, ax): #->ok
  #z score intra paziente
  #da usare quando non considero la maschera
  xz = zscore(x, axis=ax)
  return xz

def rangeNormalization(x, supLim, infLim): #->ok
  #normalizzazione nel range
  x_norm = ( (x - np.min(x)) / (np.max(x)- np.min(x)) )*(supLim - infLim) + infLim
  assert np.min(x_norm) >= infLim
  assert np.max(x_norm) <= supLim
  return x_norm

def np_imadjust(x, q1,q2): #->ok
  #applico un ehancement
  assert q1<q2
  assert q1+q2 == 1
  qq = np.quantile(x, [q1, q2])
  new = np.clip(x, qq[0], qq[1])
  return new

def multiply_mask(y, m): #->ok
  y_new = y*np.repeat(m[:,:,:,np.newaxis], 4, axis =3)
  return y_new

def convex_hull(mask): #->ok
  xx,yy,zz = np.where(mask>0)
  zz_u = np.unique(zz)
  new_mask = np.zeros(mask.shape)

  for i in zz_u:
    new_mask[:,:,i] = skm.convex_hull_image(mask[:,:,i]) 
  return new_mask

def complete_convex_hull(mask):
  new_mask = convex_hull(mask) #x y z
  new_mask = new_mask.transpose(2,0,1)   #z x y
  
  new_mask = convex_hull(new_mask)
  new_mask = new_mask.transpose(0,2,1)  #z y x
  
  new_mask = convex_hull(new_mask)
  new_mask = new_mask.transpose(2,1,0)
  return new_mask

def getFilesForSubset(basepath, list_classes, include_patient):
  ListFiles=[]
  for c in list_classes:
    listofFiles = os.listdir(basepath + '/' + c)
    for file in listofFiles:
      if include_patient(basepath + '/' + c + '/' + file):
        ListFiles.append((basepath + '/' + c + '/' + file, list_classes.index(c)))
  return ListFiles

def getListOffiles(basepath, list_classes, classe, include_patient):
  ListFiles=[]
  listofFiles = os.listdir(basepath + '/' + classe)
  for file in listofFiles:
    if include_patient(basepath + '/' + classe + '/' + file):
      ListFiles.append((basepath + '/' + classe + '/' + file, list_classes.index(classe)))
  return ListFiles, len(ListFiles)
  
class FocalLoss(nn.modules.loss._WeightedLoss):
  def __init__(self, weight=None, gamma=2,reduction='mean'):
    super(FocalLoss, self).__init__(weight,reduction=reduction)
    self.gamma = gamma
    self.weight = weight
    self.reduction = reduction

  def forward(self, input, target):
    ce_loss = F.cross_entropy(input, target,reduction='none',weight=self.weight)
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** self.gamma * ce_loss)
    if self.reduction == 'mean':
      focal_loss = focal_loss.mean()
    else:
      focal_loss = focal_loss.sum()
    return focal_loss
  
#-------------------------------------------------------------------------------> Data Augmentation
class ToTensor3D(torch.nn.Module):  #-> ok
  def __init__(self):
    super().__init__()
  
  def forward(self, tensor):
    y_new = torch.from_numpy(tensor.transpose(3,2,0,1))
    return y_new

  def __repr__(self):
    return self.__class__.__name__ + '()'

class DeleteMask(torch.nn.Module):  #-> ok
  def __init__(self):
    super().__init__()
  
  def forward(self, tensor):
    return tensor

  def __repr__(self):
    return self.__class__.__name__ + '()'

class Resize3D(torch.nn.Module):  #-> ok
  def __init__(self, size=(32,32,32)):
    self.size = size          
    super().__init__()         

  def forward(self, tensor):
    #print(tensor.shape)
    #print(tensor.unsqueeze(0).shape)
    img = F.interpolate( tensor.unsqueeze(0), self.size, align_corners =True, mode='trilinear').squeeze(0)
    #print(img.shape)
    return img
    
  def __repr__(self):
    return self.__class__.__name__ + '(size={})'.format(self.size)

#-------------------------------------------------------------------------------> Data augmentation  forma C Z X Y
#------------------------------ Rotazione n*90
class Random_Rotation(torch.nn.Module):
  def __init__(self, p=0.5, n=1):
    self.p = p                        #probabilità di effettuare la rotazione
    self.n = n                        #numero di 90 gradi           
    super().__init__()         

  def forward(self, img):
    if random() < self.p:
      img = torch.rot90(img,self.n,dims=[2,3])
    return img
    
  def __repr__(self):
    return self.__class__.__name__ + '(p={}, n={})'.format(self.p, self.n)

#------------------------------ 
class RandomZFlip(torch.nn.Module):
  def __init__(self, p=0.5):
    self.p = p                        #probabilità di effettuare il flip  
    super().__init__()                 

  def forward(self, img):

    if random() < self.p:
      img = torch.flip(img, [1])
    return img
    
  def __repr__(self):
    return self.__class__.__name__ + '(p={})'.format(self.p)
        
#------------------------------ Media e std per channel
class Normalize(torch.nn.Module):
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std
    super().__init__()
  
  def forward(self, tensor):  
    app =  tensor[0,:,:,:]
    new = ((app - self.mean[0]) /self.std[0]).unsqueeze(0)
    
    for i in range(1, tensor.shape[0]):
      app =  tensor[i,:,:,:]
      app = (app - self.mean[i]) /self.std[i]
      new = torch.cat([new, app.unsqueeze(0)], dim=0)
    return new
  
  def __repr__(self):
    return self.__class__.__name__ + '(mean={}, std={})'.format(self.mean, self.std )

#------------------------------ Media e std generica -->>ALL
class NormalizeOneMeanStd(torch.nn.Module):
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std
    super().__init__()
    
  def forward(self, tensor):
    new = (tensor - self.mean.item())/self.std.item()  
    return new
    
  def __repr__(self):
    return self.__class__.__name__ + '(mean={}, std={})'.format(self.mean, self.std)
    
#------------------------------ Bilanciamento
class BalanceConcatDataset(ConcatDataset):  
    def __init__(self, datasets):
        # bilancia il numero di campioni nei vari dataset per replicazione
        l = max([len(dataset) for dataset in datasets])
        for dataset in datasets:
            while len(dataset) < l:
                dataset.samples += sample(dataset.samples, min(len(dataset), l - len(dataset)))
        super(BalanceConcatDataset, self).__init__(datasets)
              
#-------------------------------------> Media e STD  -> numpy
def np_computeMeanAndStd_all_element(data, channel_sum, channel_sqared_sum, num_batches):
  ss = np.array(data.shape)
  channel_sum += np.sum(data)
  channel_sqared_sum  += np.sum(data**2)
  num_batches += ss.prod()  #devo fare il prodotto delle dimensioni
  return channel_sum, channel_sqared_sum, num_batches

def np_computeMeanAndStd_all(train_files):  #-> generale
  channel_sum_dce, channel_sqared_sum_dce = 0.0,0.0
  num_batches_dce = 0
  channel_sum_wat, channel_sqared_sum_wat = 0.0,0.0
  num_batches_wat = 0
  channel_sum_dwi, channel_sqared_sum_dwi = 0.0,0.0
  num_batches_dwi = 0
  
  for f,l in train_files:
    data_dce, data_water, data_dwi, cliniche = readVolume(f)
    channel_sum_dce, channel_sqared_sum_dce, num_batches_dce = np_computeMeanAndStd_all_element(data_dce, channel_sum_dce, channel_sqared_sum_dce, num_batches_dce)
    channel_sum_wat, channel_sqared_sum_wat, num_batches_wat = np_computeMeanAndStd_all_element(data_water, channel_sum_wat, channel_sqared_sum_wat, num_batches_wat)
    channel_sum_dwi, channel_sqared_sum_dwi, num_batches_dwi = np_computeMeanAndStd_all_element(data_dwi, channel_sum_dwi, channel_sqared_sum_dwi, num_batches_dwi)

  mean_dce =channel_sum_dce/num_batches_dce
  std_dce = (channel_sqared_sum_dce/num_batches_dce - mean_dce**2)**0.5
  mean_wat =channel_sum_wat/num_batches_wat
  std_wat = (channel_sqared_sum_wat/num_batches_wat - mean_wat**2)**0.5
  mean_dwi =channel_sum_dwi/num_batches_dwi
  std_dwi = (channel_sqared_sum_dwi/num_batches_dwi - mean_dwi**2)**0.5
  return mean_dce, std_dce, mean_wat, std_wat, mean_dwi, std_dwi

#-------------------------------------> 
def np_computeMeanAndStd_channnel_elemet(data, channel_sum, channel_sqared_sum, num_batches):
  ss = np.array(data.shape)
  channel_sum += np.sum(data, axis=tuple(range(0,ss.shape[0]-1)))
  channel_sqared_sum  += np.sum(data**2, axis=tuple(range(0,ss.shape[0]-1)))
  num_batches += ss[:ss.shape[0]-1].prod()
  return channel_sum, channel_sqared_sum, num_batches

def np_computeMeanAndStd_channnel(train_files):
  channel_sum_dce, channel_sqared_sum_dce = 0.0,0.0
  num_batches_dce = 0
  channel_sum_wat, channel_sqared_sum_wat = 0.0,0.0
  num_batches_wat = 0
  channel_sum_dwi, channel_sqared_sum_dwi = 0.0,0.0
  num_batches_dwi = 0
  
  for f,l in train_files:
    data_dce, data_water, data_dwi, cliniche = readVolume(f)
    channel_sum_dce, channel_sqared_sum_dce, num_batches_dce = np_computeMeanAndStd_channnel_elemet(data_dce, channel_sum_dce, channel_sqared_sum_dce, num_batches_dce)
    channel_sum_wat, channel_sqared_sum_wat, num_batches_wat = np_computeMeanAndStd_channnel_elemet(data_water, channel_sum_wat, channel_sqared_sum_wat, num_batches_wat)
    channel_sum_dwi, channel_sqared_sum_dwi, num_batches_dwi = np_computeMeanAndStd_channnel_elemet(data_dwi, channel_sum_dwi, channel_sqared_sum_dwi, num_batches_dwi)
  
  mean_dce =channel_sum_dce/num_batches_dce
  std_dce = (channel_sqared_sum_dce/num_batches_dce - mean_dce**2)**0.5
  mean_wat =channel_sum_wat/num_batches_wat
  std_wat = (channel_sqared_sum_wat/num_batches_wat - mean_wat**2)**0.5
  mean_dwi =channel_sum_dwi/num_batches_dwi
  std_dwi = (channel_sqared_sum_dwi/num_batches_dwi - mean_dwi**2)**0.5
  return mean_dce, std_dce, mean_wat, std_wat, mean_dwi, std_dwi
 
#-------------------------------------> Media e STD  -> torch
def torch_computeMeanAndStd_all_element(data, channel_sum, channel_sqared_sum, num_batches):
  data = torch.from_numpy(data)
  ss = torch.tensor(data.shape)
  channel_sum += torch.sum(data)
  channel_sqared_sum  += torch.sum(data**2)
  num_batches += ss.prod()
  return channel_sum, channel_sqared_sum, num_batches

def torch_computeMeanAndStd_all(train_files): #-> generale
  channel_sum_dce, channel_sqared_sum_dce = 0.0,0.0
  num_batches_dce = 0
  channel_sum_wat, channel_sqared_sum_wat = 0.0,0.0
  num_batches_wat = 0
  channel_sum_dwi, channel_sqared_sum_dwi = 0.0,0.0
  num_batches_dwi = 0
  
  for f,l in train_files:
    data_dce, data_water, data_dwi, cliniche = readVolume(f)
    channel_sum_dce, channel_sqared_sum_dce, num_batches_dce = torch_computeMeanAndStd_all_element(data_dce, channel_sum_dce, channel_sqared_sum_dce, num_batches_dce)
    channel_sum_wat, channel_sqared_sum_wat, num_batches_wat = torch_computeMeanAndStd_all_element(data_water, channel_sum_wat, channel_sqared_sum_wat, num_batches_wat)    
    channel_sum_dwi, channel_sqared_sum_dwi, num_batches_dwi = torch_computeMeanAndStd_all_element(data_dwi, channel_sum_dwi, channel_sqared_sum_dwi, num_batches_dwi)
  
  mean_dce =channel_sum_dce/num_batches_dce
  std_dce = (channel_sqared_sum_dce/num_batches_dce - mean_dce**2)**0.5
  mean_wat =channel_sum_wat/num_batches_wat
  std_wat = (channel_sqared_sum_wat/num_batches_wat - mean_wat**2)**0.5
  mean_dwi =channel_sum_dwi/num_batches_dwi
  std_dwi = (channel_sqared_sum_dwi/num_batches_dwi - mean_dwi**2)**0.5
  return mean_dce, std_dce, mean_wat, std_wat, mean_dwi, std_dwi
 
def torch_computeMeanAndStd_channnel_element(data, channel_sum, channel_sqared_sum, num_batches):
    data = torch.from_numpy(data)
    ss = torch.tensor(data.shape)
    channel_sum += torch.sum(data, dim=list(range(0, len(ss)-1)) )
    channel_sqared_sum  += torch.sum(data**2,  dim=list(range(0, len(ss)-1)))
    num_batches += ss[0:len(ss)-1].prod()
    return channel_sum, channel_sqared_sum, num_batches

def torch_computeMeanAndStd_channnel(train_files):
  channel_sum_dce, channel_sqared_sum_dce = 0.0,0.0
  num_batches_dce = 0
  channel_sum_wat, channel_sqared_sum_wat = 0.0,0.0
  num_batches_wat = 0
  channel_sum_dwi, channel_sqared_sum_dwi = 0.0,0.0
  num_batches_dwi = 0
  
  for f,l in train_files:
    data_dce, data_water, data_dwi, cliniche = readVolume(f)
    channel_sum_dce, channel_sqared_sum_dce, num_batches_dce = torch_computeMeanAndStd_channnel_element(data_dce, channel_sum_dce, channel_sqared_sum_dce, num_batches_dce)
    channel_sum_wat, channel_sqared_sum_wat, num_batches_wat = torch_computeMeanAndStd_channnel_element(data_water, channel_sum_wat, channel_sqared_sum_wat, num_batches_wat)
    channel_sum_dwi, channel_sqared_sum_dwi, num_batches_dwi = torch_computeMeanAndStd_channnel_element(data_dwi, channel_sum_dwi, channel_sqared_sum_dwi, num_batches_dwi)
  
  mean_dce =channel_sum_dce/num_batches_dce
  std_dce = (channel_sqared_sum_dce/num_batches_dce - mean_dce**2)**0.5
  mean_wat =channel_sum_wat/num_batches_wat
  std_wat = (channel_sqared_sum_wat/num_batches_wat - mean_wat**2)**0.5
  mean_dwi =channel_sum_dwi/num_batches_dwi
  std_dwi = (channel_sqared_sum_dwi/num_batches_dwi - mean_dwi**2)**0.5
  return mean_dce, std_dce, mean_wat, std_wat, mean_dwi, std_dwi
    
##-------------------------------------------------------------------------------> Training functions
def train_loop_validation(model_conv, 
                          trainset, Val, test, 
                          start, num_epoch, 
                          loader_opts, 
                          criterionCNN, optimizer_conv, 
                          best_acc, best_loss, best_epoca,
                          outputPath):
  
  for epochs in range(start, num_epoch + 1):
    
    TrainLoader = DataLoader(trainset, shuffle=True, **loader_opts)
    
    modelLoss_train = 0.0
    modelAcc_train = 0.0
    totalSize = 0

    model_conv.train() 
    totPred = torch.empty(0)
    totLabels = torch.empty(0)

    #-----------------------------------------------------------------------------> TRAIN      
    for inputs_dce, inputs_water, inputs_dwi, inputs_cliniche, labels in TrainLoader:
      inputs_dce = inputs_dce.type(torch.FloatTensor).cuda()
      inputs_water = inputs_water.type(torch.FloatTensor).cuda()
      inputs_dwi = inputs_dwi.type(torch.FloatTensor).cuda()
      inputs_cliniche = inputs_cliniche.type(torch.FloatTensor).cuda()
      labels = labels.cuda()
      
      optimizer_conv.zero_grad()
      model_conv.zero_grad()
       
      y = model_conv(inputs_dce, inputs_water, inputs_dwi, inputs_cliniche)
      outp, preds = torch.max(y, 1)   
      lossCNN = criterionCNN(y, labels) #media per batch
       
      lossCNN.backward()
      optimizer_conv.step()
       
      totPred = torch.cat((totPred, preds.cpu()))
      totLabels = torch.cat((totLabels, labels.cpu()))
       
      modelLoss_train += lossCNN.item() * inputs_dce.size(0)
      totalSize += inputs_dce.size(0)
      modelAcc_train += torch.sum(preds == labels.data).item()
      
    
    modelLoss_epoch_train = modelLoss_train/totalSize
    modelAcc_epoch_train  = modelAcc_train/totalSize
    
    totPred = totPred.numpy()
    totLabels = totLabels.numpy()
    acc = np.sum((totPred == totLabels).astype(int))/totalSize
    
    x = totLabels[np.where(totLabels == 1)]
    y = totPred[np.where(totLabels == 1)]
    acc_1_T = np.sum((x == y).astype(int))/x.shape[0]
    
    x = totLabels[np.where(totLabels == 0)]
    y = totPred[np.where(totLabels == 0)]
    acc_0_T = np.sum((x == y).astype(int))/y.shape[0]
    
    with open(outputPath + 'lossTrain.txt', "a") as file_object:
      file_object.write(str(modelLoss_epoch_train) +'\n')
    with open(outputPath + 'AccTrain.txt', "a") as file_object:
      file_object.write(str(modelAcc_epoch_train)+'\n')
      
    torch.save(model_conv.state_dict(), outputPath + 'train_weights.pth')
      
    #-----------------------------------------------------------------------------> VALIDATION    

    model_conv.eval()

    totalSize_val = 0
    modelLoss_val = 0.0
    modelAcc_val = 0.0

    totPred_val = torch.empty(0)
    totLabels_val = torch.empty(0)
                    
    ValLoader = DataLoader(Val, shuffle=True, **loader_opts)
    for inputs_dce, inputs_water, inputs_dwi, inputs_cliniche, labels in ValLoader:
      inputs_dce = inputs_dce.type(torch.FloatTensor).cuda()
      inputs_water = inputs_water.type(torch.FloatTensor).cuda()
      inputs_dwi = inputs_dwi.type(torch.FloatTensor).cuda()
      inputs_cliniche = inputs_cliniche.type(torch.FloatTensor).cuda()
      labels = labels.cuda()

      y = model_conv(inputs_dce, inputs_water, inputs_dwi, inputs_cliniche)
      outp, preds = torch.max(y, 1)
      lossCNN = criterionCNN(y, labels)

      totPred_val = torch.cat((totPred_val, preds.cpu()))
      totLabels_val = torch.cat((totLabels_val, labels.cpu()))

      modelLoss_val += lossCNN.item() * inputs_dce.size(0)  #Non pesata -> semplice media
      totalSize_val += inputs_dce.size(0)
      modelAcc_val += torch.sum(preds == labels.data).item()
    
        
    modelLoss_epoch_val = modelLoss_val/totalSize_val
    modelAcc_epoch_val = modelAcc_val/totalSize_val

    totPred_val = totPred_val.numpy()
    totLabels_val = totLabels_val.numpy()
    acc_val = np.sum((totPred_val == totLabels_val).astype(int))/totalSize_val
    
    x = totLabels_val[np.where(totLabels_val == 1)]
    y = totPred_val[np.where(totLabels_val == 1)]
    acc_1_V = np.sum((x == y).astype(int))/x.shape[0]
    
    x = totLabels_val[np.where(totLabels_val == 0)]
    y = totPred_val[np.where(totLabels_val == 0)]
    acc_0_v = np.sum((x == y).astype(int))/y.shape[0]
    
      
    with open(outputPath + 'lossVal.txt', "a") as file_object:
      file_object.write(str(modelLoss_epoch_val) +'\n')
    
    with open(outputPath + 'AccVal.txt', "a") as file_object:
      file_object.write(str(modelAcc_epoch_val)+'\n')
    
    with open(outputPath + 'AccVal_0.txt', "a") as file_object:
      file_object.write(str(acc_0_v)+'\n')
      
    with open(outputPath + 'AccVal_1.txt', "a") as file_object:
      file_object.write(str(acc_1_V)+'\n')

    print('[Epoch %d][TRAIN on %d [Loss: %.4f - ACC_T: %.4f - ACC_0: %.4f - ACC_1: %.4f ]][VAL on %d [Loss: %.4f - ACC_T: %.4f - ACC_0: %.4f - ACC_1: %.4f]]' 
          %(epochs, totalSize, modelLoss_epoch_train, modelAcc_epoch_train, acc_0_T, acc_1_T,
            totalSize_val, modelLoss_epoch_val, 
            modelAcc_epoch_val, acc_0_v, acc_1_V))
    
    if epochs == 1 or (modelLoss_epoch_val <= best_loss) :
      
      print('     .... Saving best weights ....')
      best_acc = modelAcc_epoch_val
      best_loss = modelLoss_epoch_val
      best_epoca = epochs
      
      #salvataggio dei migliori pesi sul validation
      torch.save(model_conv.state_dict(), outputPath + 'best_model_weights.pth')

      #vedi il test come va
      tot_size_test = 0
      model_loss_test = 0.0
      modelAcc_acc_test = 0.0
      totPred_test = torch.empty(0)
      totLabels_test = torch.empty(0)  

      TestLoader = DataLoader(test, shuffle=True, **loader_opts)
      
      for  inputs_dce, inputs_water, inputs_dwi, inputs_cliniche, labels in TestLoader:
        inputs_dce = inputs_dce.type(torch.FloatTensor).cuda()
        inputs_water = inputs_water.type(torch.FloatTensor).cuda()
        inputs_dwi = inputs_dwi.type(torch.FloatTensor).cuda()
        inputs_cliniche = inputs_cliniche.type(torch.FloatTensor).cuda()
        labels = labels.cuda()
        
        y = model_conv(inputs_dce, inputs_water, inputs_dwi, inputs_cliniche)
        outp, preds = torch.max(y, 1)
        lossCNN = criterionCNN(y, labels)

        totPred_test = torch.cat((totPred_test, preds.cpu()))
        totLabels_test = torch.cat((totLabels_test, labels.cpu()))

        model_loss_test += lossCNN.item() * inputs_dce.size(0)  #Non pesata -> semplice media
        tot_size_test += inputs_dce.size(0)
        modelAcc_acc_test += torch.sum(preds == labels.data).item()
        
      modelLoss_epoch_test = model_loss_test/tot_size_test
      modelAcc_epoch_test = modelAcc_acc_test/tot_size_test
      
      totPred_test = totPred_test.numpy()
      totLabels_test = totLabels_test.numpy()
      acc_val = np.sum((totPred_test == totLabels_test).astype(int))/tot_size_test
      
      x = totLabels_test[np.where(totLabels_test == 1)]
      y = totPred_test[np.where(totLabels_test == 1)]
      acc_1_test = np.sum((x == y).astype(int))/x.shape[0]
      
      x = totLabels_test[np.where(totLabels_test == 0)]
      y = totPred_test[np.where(totLabels_test == 0)]
      acc_0_test = np.sum((x == y).astype(int))/y.shape[0]     
      

      print('      [TEST on %d [Loss: %.4f - ACC_T: %.4f - ACC_0: %.4f - ACC_1: %.4f ]]' 
            %(tot_size_test, modelLoss_epoch_test, modelAcc_epoch_test, acc_0_test, acc_1_test))

    
    sio.savemat(outputPath + 'check_point.mat', {'best_acc': best_acc, 
                                                 'best_loss': best_loss,
                                                 'best_epoca': best_epoca,
                                                 'last_epoch': epochs})   
  return model_conv

## #-------------------------------------------------------------------------------> Predict function
def prediction_on_Test(model_conv, test, transform):
  func = nn.Softmax(dim=1)
  predicted = pd.DataFrame()
  testFiles = test.samples

  for path, label_true in testFiles:
    inputs_dce, inputs_water, inputs_dwi, inputs_cliniche= readVolume(path)

    if transform is not None:
      inputs_dce = transform[0](inputs_dce)
      inputs_water = transform[1](inputs_water)
      inputs_dwi = transform[2](inputs_dwi)
      inputs_cliniche = torch.from_numpy(inputs_cliniche)

    inputs_dce = inputs_dce.type(torch.FloatTensor).unsqueeze(0).cuda()
    inputs_water = inputs_water.type(torch.FloatTensor).unsqueeze(0).cuda()
    inputs_dwi = inputs_dwi.type(torch.FloatTensor).unsqueeze(0).cuda()
    inputs_cliniche = inputs_cliniche.type(torch.FloatTensor).unsqueeze(0).cuda()
    parti = path.split('/')[-1].split('_')
    
    y = model_conv(inputs_dce, inputs_water, inputs_dwi, inputs_cliniche)
    outp, preds = torch.max(y, 1)
    y = func(y) 
    
    for i in range(0, inputs_dce.shape[0]):
      predicted = predicted.append({'filename': path.split('/')[-1],
                                    'patient': parti[0]+'_'+parti[1],
                                    'patient_lato': parti[0]+'_'+parti[1]+'_'+parti[2],
                                    'lesion': parti[0]+'_'+parti[1]+'_'+parti[2]+'_'+parti[3]+'_'+parti[4]+'_'+parti[5].split('.')[0],
                                    'dim_lesione_dce': int(parti[5].split('.')[0]),
                                    'dim_lesione_water': int(parti[3]),
                                    'dim_lesione_dwi': int(parti[4]),
                                    'prob0': y[i,0].item(),
                                    'prob1': y[i,1].item(),
                                    'predicted': preds[i].item(),
                                    'true_class': label_true,
                                    }, ignore_index=True)
  return predicted
## #------------------------------------------------------------------------------->Retrain function
def train_loop(model_conv,
               trainset, test, 
               startEpoch, 
               num_epoch, loader_opts,
               criterionCNN, optimizer_conv, outputPath, 
               weightName, chekpointName):
  
  for epochs in range(startEpoch, num_epoch + 1):

    TrainLoader = DataLoader(trainset, shuffle=True, **loader_opts)
    modelLoss_train = 0.0
    modelAcc_train = 0.0
    totalSize = 0

    model_conv.train() 
    totPred = torch.empty(0)
    totLabels = torch.empty(0)
    #-----------------------------------------------------------------------------> TRAIN      
    for inputs_dce, inputs_water, inputs_dwi, inputs_cliniche, labels in TrainLoader:
      inputs_dce = inputs_dce.type(torch.FloatTensor).cuda()
      inputs_water = inputs_water.type(torch.FloatTensor).cuda()
      inputs_dwi = inputs_dwi.type(torch.FloatTensor).cuda()
      inputs_cliniche = inputs_cliniche.type(torch.FloatTensor).cuda()
      labels = labels.cuda()
    
      optimizer_conv.zero_grad()
      model_conv.zero_grad()
       
      y = model_conv(inputs_dce, inputs_water, inputs_dwi, inputs_cliniche)
      outp, preds = torch.max(y, 1)   
      lossCNN = criterionCNN(y, labels) #media per batch
       
      lossCNN.backward()
      optimizer_conv.step()
       
      totPred = torch.cat((totPred, preds.cpu()))
      totLabels = torch.cat((totLabels, labels.cpu()))
       
      modelLoss_train += lossCNN.item() * inputs_dce.size(0)
      totalSize += inputs_dce.size(0)
      modelAcc_train += torch.sum(preds == labels.data).item()
      
    modelLoss_epoch_train = modelLoss_train/totalSize
    modelAcc_epoch_train  = modelAcc_train/totalSize
    
    totPred = totPred.numpy()
    totLabels = totLabels.numpy()
    acc = np.sum((totPred == totLabels).astype(int))/totalSize
    
    x = totLabels[np.where(totLabels == 1)]
    y = totPred[np.where(totLabels == 1)]
    acc_1_T = np.sum((x == y).astype(int))/x.shape[0]
    
    x = totLabels[np.where(totLabels == 0)]
    y = totPred[np.where(totLabels == 0)]
    acc_0_T = np.sum((x == y).astype(int))/y.shape[0]


    torch.save(model_conv.state_dict(), outputPath + weightName)
    sio.savemat(outputPath + chekpointName, {'last_epoch': epochs}) 
      
    #-----------------------------------------------------------------------------> VALIDATION    
    model_conv.eval()
    tot_size_test = 0
    model_loss_test = 0.0
    modelAcc_acc_test = 0.0
    totPred_test = torch.empty(0)
    totLabels_test = torch.empty(0)  
    
    TestLoader = DataLoader(test, shuffle=True, **loader_opts)
    
    for  inputs_dce, inputs_water, inputs_dwi, inputs_cliniche, labels in TestLoader:
      inputs_dce = inputs_dce.type(torch.FloatTensor).cuda()
      inputs_water = inputs_water.type(torch.FloatTensor).cuda()
      inputs_dwi = inputs_dwi.type(torch.FloatTensor).cuda()
      inputs_cliniche = inputs_cliniche.type(torch.FloatTensor).cuda()
      labels = labels.cuda()

      y = model_conv(inputs_dce, inputs_water, inputs_dwi, inputs_cliniche)
      outp, preds = torch.max(y, 1)   
      lossCNN = criterionCNN(y, labels) #media per batch

      totPred_test = torch.cat((totPred_test, preds.cpu()))
      totLabels_test = torch.cat((totLabels_test, labels.cpu()))

      model_loss_test += lossCNN.item() * inputs_dce.size(0)  #Non pesata -> semplice media
      tot_size_test += inputs_dce.size(0)
      modelAcc_acc_test += torch.sum(preds == labels.data).item()
        
    modelLoss_epoch_test = model_loss_test/tot_size_test
    modelAcc_epoch_test = modelAcc_acc_test/tot_size_test

    totPred_test = totPred_test.numpy()
    totLabels_test = totLabels_test.numpy()
    acc_val = np.sum((totPred_test == totLabels_test).astype(int))/tot_size_test
      
    x = totLabels_test[np.where(totLabels_test == 1)]
    y = totPred_test[np.where(totLabels_test == 1)]
    acc_1_test = np.sum((x == y).astype(int))/x.shape[0]
      
    x = totLabels_test[np.where(totLabels_test == 0)]
    y = totPred_test[np.where(totLabels_test == 0)]
    acc_0_test = np.sum((x == y).astype(int))/y.shape[0] 

    
    print('[Epoch %d][TRAIN on %d [Loss: %.4f  ACC: %.4f - ACC_0: %.4f - ACC_1: %.4f]][TEST on %d [Loss: %.4f ][ACC_T: %.4f - ACC_0: %.4f - ACC_1: %.4f]]' 
          %(epochs, totalSize, modelLoss_epoch_train, modelAcc_epoch_train, acc_0_T, acc_1_T,
            tot_size_test, modelLoss_epoch_test, 
            modelAcc_epoch_test, acc_0_test, acc_1_test))
    
    
  return model_conv
  
#Definizione trasformazioni
def definisciTransf(meant_dce, stdt_dce, meant_water, stdt_water, meant_dwi, stdt_dwi, tipoNormalizzazione_str):
  print(tipoNormalizzazione_str)
  choice = transforms.RandomChoice([Random_Rotation(p=0.5, n=1),
                                    Random_Rotation(p=0.5, n=2), 
                                    Random_Rotation(p=0.5, n=3)])
  
  if tipoNormalizzazione_str == 'torch_computeMeanAndStd_all':
    NormFunction_dce = NormalizeOneMeanStd(meant_dce, stdt_dce)
    NormFunction_water = NormalizeOneMeanStd(meant_water, stdt_water)
    NormFunction_dwi = NormalizeOneMeanStd(meant_dwi, stdt_dwi)
  elif tipoNormalizzazione_str == 'torch_computeMeanAndStd_channnel':
    NormFunction_dce = Normalize(meant_dce, stdt_dce)
    NormFunction_water = Normalize(meant_water, stdt_water)
    NormFunction_dwi = Normalize(meant_dwi, stdt_dwi)
  else:
    NormFunction_dce = DeleteMask()
    NormFunction_water = DeleteMask()
    NormFunction_dwi =  DeleteMask()

  #----------------------------------------------------------------------------> CONTROLLO SULLA MASCHERA
  train_transform_dce = transforms.Compose([ToTensor3D(),
                                            NormFunction_dce,
                                               
                                            Resize3D(size=(64, 64, 64)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            RandomZFlip(),
                                            transforms.RandomRotation(degrees = 90),
                                            choice
                                            ])
  
  train_transform_water = transforms.Compose([ToTensor3D(),
                                            NormFunction_water,
                                               
                                            Resize3D(size=(64, 64, 64)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            RandomZFlip(),
                                            transforms.RandomRotation(degrees = 90),
                                            choice
                                            ])
  
  train_transform_dwi = transforms.Compose([ToTensor3D(),
                                            NormFunction_dwi,
                                               
                                            Resize3D(size=(64, 64, 64)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            RandomZFlip(),
                                            transforms.RandomRotation(degrees = 90),
                                            choice
                                            ])
  
  None_transform_dce  = transforms.Compose([ToTensor3D(),
                                            NormFunction_dce,
                                            Resize3D(size=(64, 64, 64))])
  
  None_transform_water  = transforms.Compose([ToTensor3D(),
                                            NormFunction_water,
                                            Resize3D(size=(64, 64, 64))])
  
  None_transform_dwi  = transforms.Compose([ToTensor3D(),
                                            NormFunction_dwi,
                                            Resize3D(size=(64, 64, 64))])
  
  #transform_cliniche = ToTensor3D() #è unico per train e val

  return [train_transform_dce, train_transform_water, train_transform_dwi],  [None_transform_dce, None_transform_water, None_transform_dwi]

def statistiche(listFiles):
  mean_dce_a, std_dce_a, mean_wat_a, std_wat_a, mean_dwi_a, std_dwi_a = np_computeMeanAndStd_all(listFiles)
  print('DCE -->')
  print(mean_dce_a)
  print(std_dce_a)
  print('WATER -->')
  print(mean_wat_a)
  print(std_wat_a)
  print('DWI -->')
  print(mean_dwi_a)
  print(std_dwi_a)
  mean_dce_c, std_dce_c, mean_wat_c, std_wat_c, mean_dwi_c, std_dwi_c = np_computeMeanAndStd_channnel(listFiles)
  print('DCE -->')
  print(mean_dce_c)
  print(std_dce_c)
  print('WATER -->')
  print(mean_wat_c)
  print(std_wat_c)
  print('DWI -->')
  print(mean_dwi_c)
  print(std_dwi_c)
  mean_dce_a_t, std_dce_a_t, mean_wat_a_t, std_wat_a_t, mean_dwi_a_t, std_dwi_a_t = torch_computeMeanAndStd_all(listFiles)
  print('DCE -->')
  print(mean_dce_a_t)
  print(std_dce_a_t)
  print('WATER -->')
  print(mean_wat_a_t)
  print(std_wat_a_t)
  print('DWI -->')
  print(mean_dwi_a_t)
  print(std_dwi_a_t)
  mean_dce_c_t, std_dce_c_t, mean_wat_c_t, std_wat_c_t, mean_dwi_c_t, std_dwi_c_t = torch_computeMeanAndStd_channnel(listFiles)
  print('DCE -->')
  print(mean_dce_c_t)
  print(std_dce_c_t)
  print('WATER -->')
  print(mean_wat_c_t)
  print(std_wat_c_t)
  print('DWI -->')
  print(mean_dwi_c_t)
  print(std_dwi_c_t)
  return mean_dce_a_t, std_dce_a_t, mean_wat_a_t, std_wat_a_t, mean_dwi_a_t, std_dwi_a_t, mean_dce_c_t, std_dce_c_t, mean_wat_c_t, std_wat_c_t, mean_dwi_c_t, std_dwi_c_t                        

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
            
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        self.conv_seg = nn.Sequential(
                                        nn.ConvTranspose3d(
                                        512 * block.expansion,
                                        32,
                                        2,
                                        stride=2
                                        ),
                                        nn.BatchNorm3d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(
                                        32,
                                        32,
                                        kernel_size=3,
                                        stride=(1, 1, 1),
                                        padding=(1, 1, 1),
                                        bias=False), 
                                        nn.BatchNorm3d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(
                                        32,
                                        num_seg_classes,
                                        kernel_size=1,
                                        stride=(1, 1, 1),
                                        bias=False) 
                                        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_seg(x)

        return x

def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

#-------------------------------------------------------------------------------> Definizione della rete  
class ReductionCoreBlock(nn.Module):
  def __init__(self, inputChannel, outchannel, ksize, stride, pad):
    super(ReductionCoreBlock, self).__init__()

    self.downsample = nn.Sequential(
        nn.Conv3d(in_channels=inputChannel, out_channels=outchannel, kernel_size = ksize, stride = stride, padding=pad, bias=False),
        nn.BatchNorm3d(outchannel),
        nn.ReLU(inplace=True),
        )
  
  def forward(self, x):
    out = self.downsample(x)
    return out

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()
  
  def forward(self, x):
    return x

class MyMedicalResNet(nn.Module):
  def __init__(self, typeResNet, weightPath, num_classes, mode):
    super().__init__()
    if typeResNet == 'resnet50':
      self.backBone = resnet50(sample_input_D = 28, sample_input_H = 28, sample_input_W=28, num_seg_classes=num_classes)
    
    elif typeResNet == 'resnet34':
      self.backBone = resnet34(sample_input_D = 28, sample_input_H = 28, sample_input_W=28, num_seg_classes=num_classes)
    
    elif typeResNet == 'resnet10':
      self.backBone = resnet10(sample_input_D = 28, sample_input_H = 28, sample_input_W=28, num_seg_classes=num_classes)
    
    elif typeResNet == 'resnet18':
      self.backBone = resnet18(sample_input_D = 28, sample_input_H = 28, sample_input_W=28, num_seg_classes=num_classes)
      
    if weightPath != '':
      print('Loading from ' + weightPath)
      self.net_dict = self.backBone.state_dict()
      pretrain = torch.load(weightPath)
      pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in self.net_dict.keys()}
      self.net_dict.update(pretrain_dict)
      self.backBone.load_state_dict(self.net_dict)
    
    if mode == 'dce':
      self.backBone.conv1 = nn.Conv3d(in_channels= 4, out_channels= 64, kernel_size= (7,7,7), stride=(2,2,2), padding=(3,3,3), bias=False)
    
    self.backBone.conv_seg = nn.AdaptiveAvgPool3d((1,1,1))
    
    if typeResNet == 'resnet50':
      self.end = nn.Sequential(
          nn.Linear(2048,num_classes)
          )
    else:
      self.end = nn.Sequential(
          nn.Linear(512,num_classes)   
          )

    
  def forward(self, x):
    x = self.backBone(x)
    x = torch.flatten(x, 1)
    x = self.end(x)
    return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CNNFusion(nn.Module):
  def __init__(self, channel_dce, channel_wat, channel_dwi, printB):  #?
    super(CNNFusion, self).__init__()
    self.printB = printB
    self.avg1 = nn.AdaptiveAvgPool3d((1,1,1))   #da B a Sb
    self.avg2 = nn.AdaptiveAvgPool3d((1,1,1))
    self.avg3 = nn.AdaptiveAvgPool3d((1,1,1))
    
    ratio = int((channel_dce + channel_wat + channel_dwi )/4)    # Cz  --> ha 3 input
    
    self.unique = nn.Sequential(                             #serve per la concatenazione
        nn.Linear(channel_dce + channel_wat + channel_dwi, ratio),
        nn.ReLU(inplace= True), 
    )

    #ne servono solo 2
    self.vector_dce_wat = nn.Sequential(
        #unisco dce wat per dwi
        nn.Linear(ratio, channel_dwi),
        nn.Sigmoid()
        
    )

    self.vector_dce_dwi = nn.Sequential(
        #unico dce dwi per wat
        nn.Linear(ratio, channel_wat ),
        nn.Sigmoid()
    )

    self.vector_wat_dwi = nn.Sequential(
        #unisco wat dwi per dce
        nn.Linear(ratio, channel_dce),
        nn.Sigmoid()
        
    )
  
  def forward(self, dce, wat, dwi):
    if self.printB:
      print(dce.shape)
      print(wat.shape)
      print(dwi.shape)

    # Sa e Sb
    dce_av = torch.flatten(self.avg1(dce),1) #vettore dce
    wat_av = torch.flatten(self.avg2(wat),1) #vettore wat
    dwi_av = torch.flatten(self.avg3(dwi),1) #vettore dwi

    if self.printB:
      print('avg pool')
      print(dce_av.shape)
      print(wat_av.shape)
      print(dwi_av.shape)

    # Z  --> concatenazione
    unique = self.unique(torch.cat((dce_av, wat_av, dwi_av), 1))

    if self.printB:
      print('unique')
      print(unique.shape)

    #sarebbe da z a Eb/Ea
    dce_wat = self.vector_dce_wat(unique) #per dwi
    dce_dwi = self.vector_dce_dwi(unique) #per wat
    wat_dwi =  self.vector_wat_dwi(unique) #per dce
    
    if self.printB:
      print('vettori ')
      print(dce_wat.shape)
      print(dce_dwi.shape)
      print(wat_dwi.shape)

    #channel wise operation  --> #B .x Eb = B_tilde (2--> foglio 1)

    dce_f = dce * wat_dwi.unsqueeze(2).unsqueeze(3).unsqueeze(4)
    wat_f = wat * dce_dwi.unsqueeze(2).unsqueeze(3).unsqueeze(4)
    dwi_f = dwi * dce_wat.unsqueeze(2).unsqueeze(3).unsqueeze(4)

    if self.printB:
      print('ris ')
      print(dce_f.shape)
      print(wat_f.shape)
      print(dwi_f.shape)

    return dce_f, wat_f, dwi_f


class My3DNet_combined(nn.Module):
  def __init__(self, typeResNet_dce, num_classes ,printB, weights_dec, weights_water, weights_dwi):
    super(My3DNet_combined, self).__init__()
    self.printB = printB
    
    self.dce_net = MyMedicalResNet(typeResNet_dce, '', num_classes, 'dce') 
    self.water_net = MyMedicalResNet(typeResNet_dce, '', num_classes, 'water')
    self.dwi_net = MyMedicalResNet(typeResNet_dce, '', num_classes, 'dwi') 
    
    #carico i pesi
    if weights_dec == '':
      print('No DCE inizialization')
    else:
      self.dce_net.load_state_dict(torch.load(weights_dec))
    
    if weights_water == '':
      print('No WATER inizialization')
    else:
      self.water_net.load_state_dict(torch.load(weights_water))

    if weights_dwi == '':
      print('No DWI inizialization')
    else:
      self.dwi_net.load_state_dict(torch.load(weights_dwi))

    #modifica della rete
    self.dce_net.end = Identity()
    self.water_net.end = Identity()
    self.dwi_net.end = Identity()

    #new layers
    self.clinic_net = nn.Sequential(
        nn.Linear(10,4),
        nn.BatchNorm1d(4),
        nn.ReLU(inplace= True),
    )

    if typeResNet_dce == 'resnet50':
      self.fusionAfter1 = CNNFusion(256,256,256, printB)
      self.fusionAfter2 = CNNFusion(512,512,512, printB)
      self.fusionAfter3 = CNNFusion(1024,1024,1024, printB)
      self.fusionAfter4 = CNNFusion(2048,2048,2048, printB)
      final_ch = 2048
    else:
      self.fusionAfter1 = CNNFusion(64,64,64, printB)
      self.fusionAfter2 = CNNFusion(128,128,128, printB)
      self.fusionAfter3 = CNNFusion(256,256,256, printB)
      self.fusionAfter4 = CNNFusion(512,512, 512, printB)
      final_ch =512

    self.last_cnn = nn.Sequential( 
        ReductionCoreBlock(inputChannel=final_ch*3, outchannel=final_ch*3, ksize=(1,1,1), stride=(1,1,1), pad=(0,0,0)),
        )

    self.avg = nn.AdaptiveAvgPool3d((1,1,1))

    self.end = nn.Sequential(
        nn.Linear(final_ch*3+4,final_ch),
        nn.ReLU(inplace= True),
        nn.Linear(final_ch,num_classes),
        )
    
  def step1(self,dce_net, water_net, dwi_net, dce_v, water_v, dwi_v):
    dce_v = dce_net.conv1(dce_v)
    dce_v = dce_net.bn1(dce_v)
    dce_v = dce_net.relu(dce_v)
    dce_v = dce_net.maxpool(dce_v)
    dce_v = dce_net.layer1(dce_v)
    
    water_v = water_net.conv1(water_v)
    water_v = water_net.bn1(water_v)
    water_v = water_net.relu(water_v)
    water_v = water_net.maxpool(water_v)
    water_v = water_net.layer1(water_v)

    dwi_v = dwi_net.conv1(dwi_v)
    dwi_v = dwi_net.bn1(dwi_v)
    dwi_v = dwi_net.relu(dwi_v)
    dwi_v = dwi_net.maxpool(dwi_v)
    dwi_v = dwi_net.layer1(dwi_v)
    return dce_v, water_v,dwi_v
    
  def step2(self,dce_net, water_net, dwi_net, dce_v, water_v,dwi_v):
   dce_v = dce_net.layer2(dce_v)
   water_v = water_net.layer2(water_v)
   dwi_v = dwi_net.layer2(dwi_v)
   return dce_v, water_v,dwi_v
   
   
  def step3(self,dce_net, water_net, dwi_net, dce_v, water_v,dwi_v):
   dce_v = dce_net.layer3(dce_v)
   water_v = water_net.layer3(water_v)
   dwi_v = dwi_net.layer3(dwi_v)
   return dce_v, water_v,dwi_v
   
  def step4(self,dce_net, water_net, dwi_net, dce_v, water_v,dwi_v):
   dce_v = dce_net.layer4(dce_v)
   water_v = water_net.layer4(water_v)
   dwi_v = dwi_net.layer4(dwi_v)
   return dce_v, water_v,dwi_v

  def forward(self, dce_v, water_v, dwi_v, feature_clnic):
    if self.printB:
      print(dce_v.shape)
      print(water_v.shape)
      print(dwi_v.shape)

    #layer 1
    dce_v, water_v, dwi_v =  self.step1(self.dce_net.backBone, self.water_net.backBone, self.dwi_net.backBone, dce_v, water_v,dwi_v)
    dce_v, water_v, dwi_v =self.fusionAfter1(dce_v,water_v, dwi_v)
    if self.printB:
      print('After1')
      print(dce_v.shape)
      print(water_v.shape)
      print(dwi_v.shape)

    #layer 2
    dce_v, water_v, dwi_v =  self.step2(self.dce_net.backBone, self.water_net.backBone, self.dwi_net.backBone, dce_v, water_v,dwi_v)
    dce_v, water_v, dwi_v =self.fusionAfter2(dce_v,water_v, dwi_v)
    if self.printB:
      print('After2')
      print(dce_v.shape)
      print(water_v.shape)
      print(dwi_v.shape)

    #layer 3
    dce_v, water_v, dwi_v =  self.step3(self.dce_net.backBone, self.water_net.backBone, self.dwi_net.backBone, dce_v, water_v,dwi_v)
    dce_v, water_v, dwi_v = self.fusionAfter3(dce_v,water_v, dwi_v)

    if self.printB:
      print('After3')
      print(dce_v.shape)
      print(water_v.shape)
      print(dwi_v.shape)

    #layer 4
    dce_v, water_v, dwi_v =  self.step4(self.dce_net.backBone, self.water_net.backBone, self.dwi_net.backBone, dce_v, water_v,dwi_v)
    dce_v, water_v, dwi_v =self.fusionAfter4(dce_v,water_v, dwi_v)

    if self.printB:
      print('After4')
      print(dce_v.shape)
      print(water_v.shape)
      print(dwi_v.shape)

    #----> Features cliniche
    if self.printB:
      print(feature_clnic.shape)
    feature_clnic = self.clinic_net(feature_clnic)
    if self.printB:
      print(feature_clnic.shape)
    
    #----> concatenazione immagini e features
    x_img = torch.cat((dce_v,water_v,dwi_v), 1)
    if self.printB:
      print(x_img.shape)

    x_img = self.last_cnn(x_img)
    if self.printB:
      print(x_img.shape)

    x_img = self.avg(x_img)  
    if self.printB:
      print(x_img.shape)

    x_img = torch.flatten(x_img,1)
    x = torch.cat((x_img, feature_clnic), 1)
    if self.printB:
      print(x.shape)
    x = self.end(x)
    return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    
    
def readVolume(path):
  x = loadmat(path)
  y_new_dce= x['dce_volume'].astype(np.float32)
  
  y_new_water= x['water_volume'].astype(np.float32)
  y_new_water = y_new_water[:,:,:,np.newaxis]

  y_new_dwi= x['dwi_volume'].astype(np.float32)
  y_new_dwi = y_new_dwi[:,:,:,np.newaxis]

  y_new_dce = rangeNormalization(y_new_dce, 1, 0)
  y_new_water = rangeNormalization(y_new_water, 1, 0)
  y_new_dwi = rangeNormalization(y_new_dwi, 1, 0)

  y_cliniche = x['features'][0,IDX_TO_CONSIDER]
  
  return y_new_dce, y_new_water, y_new_dwi, y_cliniche

#-------------------------------------------------------------------------------> DATALOADER PER IL TRAIN
class My_DatasetFolder(Dataset):
  def __init__(self, root,  transform, is_valid_file, list_classes):
    self.root = root 
    self.transform = transform
    self.is_valid_file = is_valid_file
    self.list_classes = list_classes
    self.samples = self.__get_samples()

  def __len__(self):
    return len(self.samples)

  def __get_samples(self):
    ListFiles=[]
    for c in self.list_classes:
      listofFiles = os.listdir(self.root + '/' + c)
      for file in listofFiles:
        if self.is_valid_file(self.root + '/' + c + '/' + file):
          ListFiles.append((self.root + '/' + c + '/' + file, self.list_classes.index(c)))   
    return ListFiles

  def __getitem__(self, index: int):
    path, target = self.samples[index]
    sample_dce, sample_water, sample_dwi, sample_cliniche = readVolume(path)
    if self.transform is not None:
      sample_dce = self.transform[0](sample_dce)
      sample_water = self.transform[1](sample_water)
      sample_dwi = self.transform[2](sample_dwi)
      sample_cliniche = torch.from_numpy(sample_cliniche)
      
    return sample_dce, sample_water, sample_dwi, sample_cliniche, target                                    
	
	

def main_TRAIN(fold,continue_learning, tipoNormalizzazione_str, loss, 
        basePath, classes, ch, learningRate, weightDecay, batchSize, num_epoch, 
        vali_set,test_set, esclusi, minDimLesion,
        outputPath, weight_dce, weight_water, weight_dwi, model_type):  
  print('---------------------------------> CARICAMENTO DATI')
  include_train_patient = lambda path: ((path.split('/')[-1].split('_')[0] not in vali_set + test_set) and
                                       (int(path.split('/')[-1].split('_')[-1].split('.')[0])>minDimLesion) and  
                                      (path.split('/')[-1].split('_')[0] not in esclusi))
                                      
  include_val_patient =  lambda path: ((path.split('/')[-1].split('_')[0] in vali_set) and 
                                       (int(path.split('/')[-1].split('_')[-1].split('.')[0])>minDimLesion) and  
                                       (path.split('/')[-1].split('_')[0] not in esclusi))
  
  include_test_patient =  lambda path: ((path.split('/')[-1].split('_')[0] in test_set) and 
                                       (int(path.split('/')[-1].split('_')[-1].split('.')[0])>minDimLesion) and  
                                       (path.split('/')[-1].split('_')[0] not in esclusi))

  train_files = getFilesForSubset(basePath, classes, include_train_patient)
  print(len(train_files))

  val_files = getFilesForSubset(basePath, classes, include_val_patient)
  print(len(val_files))

  test_files = getFilesForSubset(basePath, classes, include_test_patient)
  print(len(test_files))

  print(' - - - - - train')
  mean_dce_a_train, std_dce_a_train, mean_wat_a_train, std_wat_a_train, mean_dwi_a_train, std_dwi_a_train, mean_dce_c_train, std_dce_c_train, mean_wat_c_train, std_wat_c_train, mean_dwi_c_train, std_dwi_c_train  = statistiche(train_files)

  print(' - - - - - val')
  statistiche(val_files)

  print(' - - - - - test')
  statistiche(test_files)

  if tipoNormalizzazione_str == 'torch_computeMeanAndStd_channnel':
    train_transform_vett, None_transform_vett = definisciTransf(mean_dce_c_train, std_dce_c_train, mean_wat_c_train, std_wat_c_train, mean_dwi_c_train, std_dwi_c_train, tipoNormalizzazione_str)
  else:
   train_transform_vett, None_transform_vett = definisciTransf(mean_dce_a_train, std_dce_a_train, mean_wat_a_train, std_wat_a_train, mean_dwi_a_train, std_dwi_a_train, tipoNormalizzazione_str)                                    

  print(train_transform_vett)   
  print(None_transform_vett)     


  Trainset = [] 
  for c in classes:
    print(' Loading ' + c)
    is_valid_class = lambda path: c == path.split('/')[-2]
    check_file = lambda path: include_train_patient(path) and is_valid_class(path)
    Trainset.append(My_DatasetFolder(root = basePath, transform= train_transform_vett, is_valid_file=check_file, list_classes=classes ))


  print('0_cavo elements ', str(len(Trainset[0].samples)))
  print('1_cavo elements ', str(len(Trainset[1].samples)))
  
  num_1 = len(Trainset[1].samples)
  num_0 = len(Trainset[0].samples)
  
  completeTrainSet = BalanceConcatDataset(Trainset)
  print('0_cavo elements ', str(len(completeTrainSet.datasets[0].samples)))
  print('1_cavo elements ', str(len(completeTrainSet.datasets[1].samples)))

  Val  = My_DatasetFolder(root = basePath, transform=None_transform_vett, is_valid_file=include_val_patient, list_classes=classes)
  Test = My_DatasetFolder(root = basePath, transform=None_transform_vett, is_valid_file=include_test_patient, list_classes=classes)
  print('Validation ',str(len(Val.samples)))
  print('Test ', str(len(Test.samples)))
  
  print('---------------------------------> ADDESTRAMENTO')
  
  #definizione del modello
  #typeResNet_dce, num_classes ,printB, weights_dec, weights_water, weights_dwi
  model_conv = My3DNet_combined(model_type,2, False, weight_dce, weight_water, weight_dwi)
  model_conv = model_conv.cuda() 
  print(count_parameters(model_conv))

  optimizer_conv = optim.Adam(model_conv.parameters(), lr=learningRate, weight_decay= weightDecay)
  if loss == 'focal':
    criterionCNN = FocalLoss()
  else:
    criterionCNN = nn.CrossEntropyLoss()

  loader_opts = {'batch_size': batchSize, 'num_workers': 0, 'pin_memory': False}
  print('     Before Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))

  if not continue_learning:
    #inizializzazione senza check point
    best_acc = 0.0   
    best_loss = 0.0 
    best_epoca = 0
    startEpoch = 1
  else:
    print('RELOAD')
    stato = sio.loadmat(outputPath + 'check_point.mat')
    best_acc = stato['best_acc'][0][0]
    best_loss = stato['best_loss'][0][0]
    best_epoca = stato['best_epoca'][0][0]
    startEpoch = stato['last_epoch'][0][0] + 1
    model_conv.load_state_dict(torch.load(outputPath + 'train_weights.pth'))

  model_conv = train_loop_validation(model_conv, 
                                     completeTrainSet, Val, Test, 
                                     startEpoch, num_epoch, 
                                     loader_opts, 
                                     criterionCNN, optimizer_conv, 
                                     best_acc, best_loss, best_epoca, 
                                     outputPath)

  print('     After Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))
  model_conv.cpu()
  del model_conv
  print('     After Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))
  
  print('---------------------------------> BEST MODEL')
  lossModel_Train = []
  lossModel_val = []
  lossModel_val_weighted = []
  
  accModel_Train = []
  accModel_val = []

  Acc_0 = []
  Acc_1 = []

  file = open(outputPath + 'lossTrain.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    lossModel_Train.append(float(element))

  file = open(outputPath + 'lossVal.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    lossModel_val.append(float(element))

  plt.figure()
  plt.title("Model: Training Vs Validation Losses")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.plot(list(range(1,len(lossModel_Train)+1)), lossModel_Train, color='r', label="Training Loss")  
  plt.plot(list(range(1, len(lossModel_val)+1)), lossModel_val, color='g', label="Validation Loss")
  plt.legend()
  plt.savefig(outputPath + 'LossTrainVal.png')


  file = open(outputPath + 'AccTrain.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    accModel_Train.append(float(element))

  file = open(outputPath + 'AccVal.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    accModel_val.append(float(element))

  plt.figure()
  plt.title("Training Vs Validation Accuracies")
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(list(range(1, len(accModel_Train)+1)), accModel_Train, color='r', label="Training Accuracy")
  plt.plot(list(range(1, len(accModel_val)+1)), accModel_val, color='g', label="Validation Accuracy")
  plt.legend()
  plt.savefig(outputPath + 'AccTrainVal.png')
  
  
  file = open(outputPath + 'AccVal_0.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    Acc_0.append(float(element))

  file = open(outputPath + 'AccVal_1.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    Acc_1.append(float(element))

  plt.figure()
  plt.title("Validation Accuracies")
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(list(range(1, len(Acc_0)+1)), Acc_0, color='r', label="Val Accuracy class 0")
  plt.plot(list(range(1, len(Acc_1)+1)), Acc_1, color='g', label="Val Accuracy class 1")
  plt.plot(list(range(1, len(accModel_val)+1)), accModel_val, color='b', label="Total Val Accuracy")
  plt.legend()
  plt.savefig(outputPath + 'AccVal.png')
  
  #definizione del modello
  model_conv = My3DNet_combined(model_type,2, False, weight_dce, weight_water, weight_dwi)
  print(count_parameters(model_conv))
  model_conv.load_state_dict(torch.load(outputPath + 'best_model_weights.pth'))
  model_conv = model_conv.cuda() 

  model_conv.eval()
  #model_conv, test, transform
  tabella = prediction_on_Test(model_conv, Test, None_transform_vett)
  tabella.to_csv(outputPath + 'TabellaBestModel.csv', sep = ',', index=False)


  accuracy = np.sum(tabella.true_class.values == tabella.predicted.values)/tabella.shape[0]
  t0 = tabella[tabella.true_class == 0]
  t1 = tabella[tabella.true_class == 1]

  accuracy_0 = np.sum(t0.true_class.values == t0.predicted.values)/t0.shape[0]
  accuracy_1 = np.sum(t1.true_class.values == t1.predicted.values)/t1.shape[0]

  print('Accuracy')
  print(accuracy)
  print('Accuracy_0')
  print(accuracy_0)
  print('Accuracy_1')
  print(accuracy_1)

  model_conv.cpu()
  del model_conv
  print('     After Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))
  
     
def main_validation(fold,continue_fine_tuning, tipoNormalizzazione_str, loss,
     basePath, classes, ch, learningRate, weightDecay, batchSize, epocheFineTuning, fattore,
     vali_set, test_set, esclusi, minDimLesion,
     outputPath, weight_dce, weight_water, weight_dwi, model_type):
     
  include_train_patient = lambda path: ((path.split('/')[-1].split('_')[0] not in vali_set + test_set) and 
                                        (int(path.split('/')[-1].split('_')[-1].split('.')[0])>minDimLesion) and  
                                        (path.split('/')[-1].split('_')[0] not in esclusi))


  train_files = getFilesForSubset(basePath, classes, include_train_patient)
  print(len(train_files))
  print(' - - - - - train')
  mean_dce_a_train, std_dce_a_train, mean_wat_a_train, std_wat_a_train, mean_dwi_a_train, std_dwi_a_train, mean_dce_c_train, std_dce_c_train, mean_wat_c_train, std_wat_c_train, mean_dwi_c_train, std_dwi_c_train  = statistiche(train_files)


  if tipoNormalizzazione_str == 'torch_computeMeanAndStd_channnel':
    train_transform_vett, None_transform_vett = definisciTransf(mean_dce_c_train, std_dce_c_train, mean_wat_c_train, std_wat_c_train, mean_dwi_c_train, std_dwi_c_train, tipoNormalizzazione_str)
  else:
   train_transform_vett, None_transform_vett = definisciTransf(mean_dce_a_train, std_dce_a_train, mean_wat_a_train, std_wat_a_train, mean_dwi_a_train, std_dwi_a_train, tipoNormalizzazione_str)                                    

  print(train_transform_vett)   
  print(None_transform_vett) 

  #-----------------------------------------------------------------------------> fine tuning finale
  include_train_patient_fine = lambda path: ((path.split('/')[-1].split('_')[0] in vali_set) and 
                                        (int(path.split('/')[-1].split('_')[-1].split('.')[0])>minDimLesion) and  
                                        (path.split('/')[-1].split('_')[0] not in esclusi))
  

  include_test_patient =  lambda path: ((path.split('/')[-1].split('_')[0] in test_set) and 
                                         (int(path.split('/')[-1].split('_')[-1].split('.')[0])>minDimLesion) and  
                                         (path.split('/')[-1].split('_')[0] not in esclusi))

  Trainset = []   
  for c in classes:
    print(' Loading ' + c)
    is_valid_class = lambda path: c == path.split('/')[-2]
    check_file = lambda path: include_train_patient_fine(path) and is_valid_class(path)
    Trainset.append(My_DatasetFolder(root = basePath, transform=train_transform_vett, is_valid_file=check_file, list_classes=classes))
  print('0_cavo elements ', str(len(Trainset[0].samples)))
  print('1_cavo elements ', str(len(Trainset[1].samples)))
  completeTrainSet = BalanceConcatDataset(Trainset)
  print('0_cavo elements ', str(len(completeTrainSet.datasets[0].samples)))
  print('1_cavo elements ', str(len(completeTrainSet.datasets[1].samples)))

  Test = My_DatasetFolder(root = basePath, transform=None_transform_vett, is_valid_file=include_test_patient, list_classes=classes)
  print('Test ', str(len(Test.samples)))

  #-------------------------------------------------------------------------------> Definizione del modello
  #definizione del modello
  model_conv = My3DNet_combined(model_type,2, False, weight_dce, weight_water, weight_dwi)
  model_conv.load_state_dict(torch.load(outputPath + 'best_model_weights.pth'))  
  model_conv = model_conv.cuda() 
  print(count_parameters(model_conv))

  optimizer_conv = optim.Adam(model_conv.parameters(), lr=learningRate*fattore, weight_decay= weightDecay)
  if loss == 'focal':
    criterionCNN = FocalLoss()
  else:
    criterionCNN = nn.CrossEntropyLoss()
  loader_opts = {'batch_size': batchSize, 'num_workers': 0, 'pin_memory': False}

  print('     Before Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))
  startEpoch = 1

  if continue_fine_tuning:
    print('RELOAD')
    stato = sio.loadmat(outputPath + 'check_point_for_fineTuning.mat')
    startEpoch = stato['last_epoch'][0][0] + 1
    model_conv.load_state_dict(torch.load(outputPath + 'FinalWeightsFineTuning.pth'))

  #------------------------------------------------------------ Addestramento
  model_conv = train_loop(model_conv,
                          completeTrainSet, Test,
                          startEpoch,
                          epocheFineTuning, loader_opts, 
                          criterionCNN, optimizer_conv, outputPath, 'FinalWeightsFineTuning.pth', 'check_point_for_fineTuning.mat')

  model_conv.cpu()
  del model_conv
  print('     After Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))

  #definizione del modello
  model_conv = My3DNet_combined(model_type,2, False, weight_dce, weight_water, weight_dwi)
  model_conv.load_state_dict(torch.load(outputPath + 'FinalWeightsFineTuning.pth'))
  model_conv = model_conv.cuda() 
  print(count_parameters(model_conv))


  model_conv.eval()
  tabella = prediction_on_Test(model_conv, Test, None_transform_vett)
  tabella.to_csv(outputPath + 'TabellaFinalModelFineTuning.csv', sep = ',', index=False)


  accuracy = np.sum(tabella.true_class.values == tabella.predicted.values)/tabella.shape[0]
  t0 = tabella[tabella.true_class == 0]
  t1 = tabella[tabella.true_class == 1]

  accuracy_0 = np.sum(t0.true_class.values == t0.predicted.values)/t0.shape[0]
  accuracy_1 = np.sum(t1.true_class.values == t1.predicted.values)/t1.shape[0]

  print('Accuracy')
  print(accuracy)
  print('Accuracy_0')
  print(accuracy_0)
  print('Accuracy_1')
  print(accuracy_1)

  model_conv.cpu()
  del model_conv
  print('     After Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))


def main_final_restrain(fold,continue_learning_retrain, tipoNormalizzazione_str, loss,
     basePath, classes, ch, learningRate, weightDecay, batchSize, 
     vali_set, test_set, esclusi, minDimLesion,
     outputPath, weight_dce, weight_water, weight_dwi, model_type):
     
  include_train_patient_fine = lambda path: ((path.split('/')[-1].split('_')[0] not in test_set) and 
                                        (int(path.split('/')[-1].split('_')[-1].split('.')[0])>minDimLesion) and  
                                        (path.split('/')[-1].split('_')[0] not in esclusi))
  
  include_test_patient =  lambda path: ((path.split('/')[-1].split('_')[0] in test_set) and 
                                         (int(path.split('/')[-1].split('_')[-1].split('.')[0])>minDimLesion) and  
                                         (path.split('/')[-1].split('_')[0] not in esclusi))

  train_files = getFilesForSubset(basePath, classes, include_train_patient_fine)
  print(len(train_files))
  
  print(' - - - - - train')
  mean_dce_a_train, std_dce_a_train, mean_wat_a_train, std_wat_a_train, mean_dwi_a_train, std_dwi_a_train, mean_dce_c_train, std_dce_c_train, mean_wat_c_train, std_wat_c_train, mean_dwi_c_train, std_dwi_c_train  = statistiche(train_files)


  if tipoNormalizzazione_str == 'torch_computeMeanAndStd_channnel':
    train_transform_vett, None_transform_vett = definisciTransf(mean_dce_c_train, std_dce_c_train, mean_wat_c_train, std_wat_c_train, mean_dwi_c_train, std_dwi_c_train, tipoNormalizzazione_str)
  else:
   train_transform_vett, None_transform_vett = definisciTransf(mean_dce_a_train, std_dce_a_train, mean_wat_a_train, std_wat_a_train, mean_dwi_a_train, std_dwi_a_train, tipoNormalizzazione_str)                                    

  print(train_transform_vett)   
  print(None_transform_vett) 
  #----------------------------------------------------> fine tuning finale

  Trainset = []   
  for c in classes:
    print(' Loading ' + c)
    is_valid_class = lambda path: c == path.split('/')[-2]
    check_file = lambda path: include_train_patient_fine(path) and is_valid_class(path)
    Trainset.append(My_DatasetFolder(root = basePath, transform=train_transform_vett, is_valid_file=check_file, list_classes=classes))
  print('0_cavo elements ', str(len(Trainset[0].samples)))
  print('1_cavo elements ', str(len(Trainset[1].samples)))
  completeTrainSet = BalanceConcatDataset(Trainset)
  print('0_cavo elements ', str(len(completeTrainSet.datasets[0].samples)))
  print('1_cavo elements ', str(len(completeTrainSet.datasets[1].samples)))

  Test = My_DatasetFolder(root = basePath, transform=None_transform_vett, is_valid_file=include_test_patient, list_classes=classes)
  print('Test ', str(len(Test.samples)))

  #-------------------------------------------------------------------------------> Definizione del modello
  #definizione del modello
  model_conv = My3DNet_combined(model_type,2, False, weight_dce, weight_water, weight_dwi)
  model_conv = model_conv.cuda() 
  print(count_parameters(model_conv))

  optimizer_conv = optim.Adam(model_conv.parameters(), lr=learningRate, weight_decay= weightDecay)
  if loss == 'focal':
    criterionCNN = FocalLoss()
  else:
    criterionCNN = nn.CrossEntropyLoss()
  loader_opts = {'batch_size': batchSize, 'num_workers': 0, 'pin_memory': False}

  print('     Before Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))
  stato = sio.loadmat(outputPath + 'check_point.mat')
  best_epoca_onMinLoss = stato['best_epoca'][0][0]

  accModel_val = []
  file = open(outputPath + 'AccVal.txt', 'r')
  Testo = file.readlines()
  for element in Testo:
    accModel_val.append(float(element))
  
  best_epoca_onMax = np.argmax(accModel_val) +1
  
  print(best_epoca_onMax)
  best_epoca = np.max([best_epoca_onMinLoss, best_epoca_onMax])

  print('Retrain for ' + str(best_epoca) + ' epochs')
  startEpoch = 1
  #best_epoca = 1
  if continue_learning_retrain:
    print('RELOAD')
    stato = sio.loadmat(outputPath + 'check_point_for_retrain.mat')
    startEpoch = stato['last_epoch'][0][0] + 1
    model_conv.load_state_dict(torch.load(outputPath + 'FinalWeights.pth'))

  #------------------------------------------------------------ Addestramento
  model_conv = train_loop(model_conv,
                        completeTrainSet, Test, 
                        startEpoch, 
                        best_epoca, loader_opts,
                        criterionCNN, optimizer_conv, outputPath, 'FinalWeights.pth', 'check_point_for_retrain.mat')

  model_conv.cpu()
  del model_conv
  print('     After Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))

  #definizione del modello
  model_conv = My3DNet_combined(model_type,2, False, weight_dce, weight_water, weight_dwi)
  model_conv.load_state_dict(torch.load(outputPath + 'FinalWeights.pth'))
  model_conv = model_conv.cuda() 
  print(count_parameters(model_conv))


  model_conv.eval()
  tabella = prediction_on_Test(model_conv, Test, None_transform_vett)
  tabella.to_csv(outputPath + 'TabellaFinalModel.csv', sep = ',', index=False)


  accuracy = np.sum(tabella.true_class.values == tabella.predicted.values)/tabella.shape[0]
  t0 = tabella[tabella.true_class == 0]
  t1 = tabella[tabella.true_class == 1]

  accuracy_0 = np.sum(t0.true_class.values == t0.predicted.values)/t0.shape[0]
  accuracy_1 = np.sum(t1.true_class.values == t1.predicted.values)/t1.shape[0]

  print('Accuracy')
  print(accuracy)
  print('Accuracy_0')
  print(accuracy_0)
  print('Accuracy_1')
  print(accuracy_1)

  model_conv.cpu()
  del model_conv
  print('     After Training: GPU Memory  %d bytes'%(torch.cuda.memory_allocated()))