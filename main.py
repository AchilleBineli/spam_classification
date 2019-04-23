#######import models needed############
import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
import torch.nn as nn
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
import texttable as tt
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from string import punctuation
import seaborn as sns; sns.set()
from sklearn import svm
from keras.layers import  Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import models.lstm
import models.rcnn
import models.cnn
import models.svm

from data_visualization import data_visualization
from load_data import load_dataset









#########################################TRAIN FUNCTIONS##################################################################################
#############################################################################################################################################################################################################################

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for para in params:
        para.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_batch, epoch):
    all_epoch_loss = 0
    all_epoch_acc = 0

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_batch):
        text = batch.Text[0]
        target = batch.Label
        target = torch.autograd.Variable(target).long()
        if (text.size()[0] is not 32):
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
        all_epoch_loss += loss.item()
        all_epoch_acc += acc.item()
        
    return all_epoch_loss/len(train_batch), all_epoch_acc/len(train_batch)

def eval_model(model, val_iter):
    all_epoch_loss = 0
    all_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.Text[0]
            if (text.size()[0] is not 32):
                continue
            target = batch.Label
            target = torch.autograd.Variable(target).long()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            all_epoch_loss += loss.item()
            all_epoch_acc += acc.item()

    return all_epoch_loss/len(val_iter), all_epoch_acc/len(val_iter)
    

learning_rate = 2e-5
batch_size = 32
output_size = 2
hidden_size = 256
embedding_length = 300


###################EXPERIMENTS#################

def experiment(model, epochs=3):

    """
    Train each model

    Arguments:
    model -- a class or a function of training model

    Return:
    para -- a dict that includes:
        all_train_loss -- a list that contains the loss of train of each epoch
        all_train_acc -- a list that contains the tarin accuracy  of each epoch
        all_val_loss -- a list that contains the validation loss of each epoch
        all_val_acc -- a list that ocntains the validation accuracy of each epoch
        time needed -- the time that every model consums
        test_loss -- the loss of the test dataset
        test_acc -- the accuracy of the test dataset
    """

    para = {}

    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []
    time_needed = None
    
    if model is "LSTMClassifier":
        para = LSTMClassifer2(epochs=epochs)    
    else:
        debut = time.time()
        model = model(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
        loss_fn = F.cross_entropy
        for epoch in range(epochs):
            train_loss, train_acc = train_model(model, train_iter, epoch)
            val_loss, val_acc = eval_model(model, valid_iter)

            all_train_loss.append(train_loss)
            all_train_acc.append(train_acc)
            all_val_loss.append(val_loss)
            all_val_acc.append(val_acc)

            print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

        test_loss, test_acc = eval_model(model, test_iter)
        print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
        fini =time.time() 
        time_needed = fini - debut

        para["all_train_loss"] = all_train_loss
        para['all_train_acc'] = all_train_acc
        para['all_val_loss'] = all_val_loss
        para['all_val_acc'] = all_val_acc
        para['time_needed'] = time_needed
        para['test_loss'] = test_loss
        para['test_acc'] = test_acc

    return para



################### MAIN FUNCTION ###################


################# SVM ##############################
svm(data)

####################DEEP LEARNING ###################

TEXT, vocab_size, word_embeddings, train_batch, valid_batch, test_batch = load_dataset()
photo_para = {}
model_list = {}
model_list['LSTMClassifier'] = LSTMClassifier
model_list['RCNN'] = RCNN
model_list["CNN"] = CNN

loss_fn = F.cross_entropy
for model in model_list:
    para = experiment(model_list[model],epochs=2)
    photo_para[model] = para

##########################################TABLE COMPARING DIFFERENT MODELS###############################
#########################################################################################################
tab = tt.Texttable() 
headings = ['Names','LOSS','ACCURACY',"TIME(seconds)"]
tab.header(headings) 

for para in photo_para:
    row = [para,photo_para[para]["test_loss"],photo_para[para]["test_acc"],photo_para[para]["time_needed"]]
    tab.add_row(row)
s = tab.draw()
print(s)  



############################## the comparazition of precision of different models ##############
############################################################################################
fig = plt.figure()
plt.title("the comparazition of precision of different models")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

for para in photo_para:
    plt.xticks(np.arange(1,len(photo_para[para]["all_train_acc"])+1),1)
    plt.plot(np.arange(1,len(photo_para[para]["all_train_acc"])+1),photo_para[para]["all_train_acc"],label=para)

plt.legend()
plt.show()
