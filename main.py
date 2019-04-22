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


################### DATA VISUALISATON ###############

def data_visualization():

    data = pd.read_table('smsspamcollection/SMSSpamCollection',
                       sep='\t',
                       header=None,
                       names=['label', 'sms_message'])
    #the primary information of the data set
    print(data.head())
    print()
    print()
    
    
    
    #the proportion of spam and ham
    data.columns = ["category", "text"]
    colors = ['yellowgreen', 'lightcoral']
    data["category"].value_counts().plot(kind = 'pie', colors=colors, explode = [0, 0.1], figsize = (8, 8), autopct = '%1.1f%%', shadow = True)
    plt.ylabel("Spam vs Ham")
    plt.legend(["Ham", "Spam"])
    plt.show()
    
    print()
    print()
    
    #the number of spam and ham
    data.columns = ["category", "text"]
    colors = ['yellowgreen', 'lightcoral']
    data['category'].value_counts().plot(kind = 'bar', colors=colors)
    plt.ylabel("Spam vs Ham")
    plt.show()
    
    print()
    print()
    
    
    
    # plotting graph by length.
    ham =data[data['category'] == 'ham']['text'].str.len()
    sns.distplot(ham, label='Ham')
    spam = data[data['category'] == 'spam']['text'].str.len()
    sns.distplot(spam, label='Spam')
    plt.title('Distribution by Length')
    plt.legend
    plt.show()
    
    
    def ponctuation (str):
        for p in list(punctuation):
            str = str.lower().replace(p, '')
        return str
    data['text'] = data.text.apply(ponctuation)
    def nb_word (str):
        return len(str.split())
    data['nb_word'] = data.text.apply(nb_word)
    def nb_char (str):
        return len(list(str))
    data['nb_char'] = data.text.apply(nb_char)
    print(data.head())
    
    ####the probability of the nb word###
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(2 ,1,1)
    ax1.hist(data['nb_word'], normed=True, bins=60, color='black', alpha=0.5)
    plt.xlim(0,65)
    plt.xlabel('nb word')
    plt.ylabel('Probability')
    ax2 = fig.add_subplot(2, 1, 2)
    bins = np.histogram(np.hstack((data.loc[data.category=='ham']['nb_word'], data.loc[data.category=='spam']['nb_word'])), bins=70)[1]
    plt.hist(data.loc[data.category=='ham']['nb_word'], bins, normed=True, color='yellowgreen', alpha=0.8, label='ham')
    plt.hist(data.loc[data.category=='spam']['nb_word'], bins, normed=True, color='lightcoral', alpha=0.8, label='spam')
    plt.legend(loc='upper right')
    plt.xlim(0, 60)
    plt.xlabel('Nb word')
    plt.ylabel('Probability')
    plt.title('Ham vs Spam')
    plt.subplots_adjust(hspace=0.5)
    plt.show
    
    
    ####the probability of the nb char
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(2 ,1,1)
    ax1.hist(data['nb_char'], normed=True, bins=60, color='black', alpha=0.5)
    plt.xlim(0,300)
    plt.xlabel('nb Char')
    plt.ylabel('Probability')
    ax2 = fig.add_subplot(2, 1, 2)
    bins = np.histogram(np.hstack((data.loc[data.category=='ham']['nb_char'], data.loc[data.category=='spam']['nb_char'])), bins=70)[1]
    plt.hist(data.loc[data.category=='ham']['nb_char'], bins, normed=True, color='yellowgreen', alpha=0.8, label='ham')
    plt.hist(data.loc[data.category=='spam']['nb_char'], bins, normed=True, color= 'lightcoral', alpha=0.8, label='spam')
    plt.legend(loc='upper right')
    plt.xlim(0, 300)
    plt.xlabel('Nb char')
    plt.ylabel('Probability')
    plt.title('Ham vs Spam')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    
    

###############LOAD DATA FOR TRAINING##########
def load_dataset(test_sen=None):
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
    LABEL = data.LabelField()
    train_data,test_data = data.TabularDataset.splits(
        skip_header=True,
        path='/Users/huyongbing/Desktop/poly/deep/', train ='train.csv',test='test.csv' ,format='csv',
        fields=[('Label', LABEL),('Text', TEXT)])
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)
    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))
    train_data, valid_data = train_data.split() 
    train_batch, valid_batch, test_batch = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32, sort_key=lambda x: len(x.Text), repeat=False, shuffle=True)
    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_batch, valid_batch, test_batch

################################################SVM#####################

def svm(data):

    texts = []
    labels = []
    for i, label in enumerate(data['Category']):
        texts.append(data['Message'][i])
        if label == 'ham':
            labels.append(0)
        else:
            labels.append(1)

    texts = np.asarray(texts)
    labels = np.asarray(labels)


    max_features = 10000
    maxlen = 300

    training_samples = int(5572 * .8)
    validation_samples = int(5572 - training_samples)
    print(len(texts) == (training_samples + validation_samples))
    print("The number of training {0}, validation {1} ".format(training_samples, validation_samples))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print("Found {0} unique words: ".format(len(word_index)))

    data = pad_sequences(sequences, maxlen=maxlen)

    print("data shape: ", data.shape)

    np.random.seed(42)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    ####divide data to train and test
    texts_train = data[:training_samples]
    y_train = labels[:training_samples]
    texts_test = data[training_samples:]
    y_test = labels[training_samples:]

    clf = svm.SVC(gamma='auto',kernel='linear')
    clf.fit(texts_train,y_train)
    clf.score(texts_test,y_test)


############ MODULES ####################

class LSTMClassifier(nn.Module):
    
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(LSTMClassifier, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, input_sentence, batch_size=None):
        input = self.word_embeddings(input_sentence)
        input = input.permute(1, 0, 2)
        if batch_size is None:
            h_first = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
            c_first = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        else:
            h_first = Variable(torch.zeros(1, batch_size, self.hidden_size))
            c_first = Variable(torch.zeros(1, batch_size, self.hidden_size))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_first, c_first))
        output = self.dropout_layer(final_hidden_state[-1])
        output = self.hidden2out(output)
        final_output = self.softmax(output)
        return final_output

class RCNN(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(RCNN, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) 
        self.dropout = 0.5
        self.lstm = nn.LSTM(embedding_length, hidden_size, dropout=self.dropout, bidirectional=True)
        self.W2 = nn.Linear(3*hidden_size+embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sentence, batch_size=None):
        input = self.word_embeddings(input_sentence) 
        input = input.permute(1, 0, 2) 
        if batch_size is None:
            h_first = Variable(torch.zeros(2, self.batch_size, self.hidden_size)) 
            c_first = Variable(torch.zeros(2, self.batch_size, self.hidden_size)) 
        else:
            h_first = Variable(torch.zeros(2, batch_size, self.hidden_size))
            c_first = Variable(torch.zeros(2, batch_size, self.hidden_size))

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_first, c_first))
        
        last_code = torch.cat((output, input), 2).permute(1, 0, 2)
        y = self.W2(last_code) 
        y = y.permute(0, 2, 1) 
        y = F.max_pool1d(y, y.size()[2]) 
        y = y.squeeze(2)
        logits = self.label(y)
        
        return logits


class CNN(nn.Module):
    def __init__(self, batch_size, output_size,hidden_size,vocab_size, embedding_length, weights,in_channels=1,out_channels=3,kernel_heights=[1,2,3],stride=1,padding=0,keep_probab=0.1):
        super(CNN, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.stride = stride
        self.padding = padding
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length), stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length), stride, padding)
        self.dropout = nn.Dropout(keep_probab)
        self.label = nn.Linear(len(kernel_heights)*out_channels, output_size)

    def conv_block(self, input, conv_layer):
        conv_output = conv_layer(input)
        act = F.relu(conv_output.squeeze(3))
        max_output = F.max_pool1d(act, act.size()[2]).squeeze(2)

        return max_output

    def forward(self, input_sentences, batch_size=None):

       

        input = self.word_embeddings(input_sentences)

        input = input.unsqueeze(1)

        max_output1 = self.conv_block(input, self.conv1)
        max_output2 = self.conv_block(input, self.conv2)
        max_output3 = self.conv_block(input, self.conv3)

        all_out = torch.cat((max_output1, max_output2, max_output3), 1)
        fc_in = self.dropout(all_out)
        logits = self.label(fc_in)

        return logits


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


###############EXPERIMENTS#################

def experiment(model, epochs=1):

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
    
    debut = time.time()
    
    model = model(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    loss_fn = F.cross_entropy
    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, train_batch, epoch)
        val_loss, val_acc = eval_model(model, valid_batch)

        all_train_loss.append(train_loss)
        all_train_acc.append(train_acc)
        all_val_loss.append(val_loss)
        all_val_acc.append(val_acc)

        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

    test_loss, test_acc = eval_model(model, test_batch)
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


####### DATA VISUALISATON ######
data_visualization()

################# SVM ##############################
####################################################
####################################################
svm(data)


####################DEEP LEARNING ###################
####################################################
####################################################
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
