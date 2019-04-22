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
from numpy.random import RandomState

###############DATA FUNCTION FOR HANDING DATA##########
def load_dataset(test_sen=None):
    df = pd.read_csv('./data/spam.csv', encoding= "ISO-8859-1")
    rng = RandomState()

    train = df.sample(frac=0.7, random_state=rng)
    test = df.loc[~df.index.isin(train.index)]

    train.to_csv('./data/train.csv', header=False, index=False) 
    test.to_csv('./data/test.csv', header=False, index=False)
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
    LABEL = data.LabelField()
    train_data,test_data = data.TabularDataset.splits(
        skip_header=True,
        path='./data', train ='train.csv',test='test.csv' ,format='csv',
        fields=[('Label', LABEL),('Text', TEXT)])
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)
    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))
    train_data, valid_data = train_data.split() # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32, sort_key=lambda x: len(x.Text), repeat=False, shuffle=True)
    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter


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
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
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
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.dropout = 0.8
        self.lstm = nn.LSTM(embedding_length, hidden_size, dropout=self.dropout, bidirectional=True)
        self.W2 = nn.Linear(2*hidden_size+embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sentence, batch_size=None):
        input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences, embedding_length)
        input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
        if batch_size is None:
            h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size)) # Initial hidden state of the LSTM
            c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size)) # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size))

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        
        final_encoding = torch.cat((output, input), 2).permute(1, 0, 2)
        y = self.W2(final_encoding) # y.size() = (batch_size, num_sequences, hidden_size)
        y = y.permute(0, 2, 1) # y.size() = (batch_size, hidden_size, num_sequences)
        y = F.max_pool1d(y, y.size()[2]) # y.size() = (batch_size, hidden_size, 1)
        y = y.squeeze(2)
        logits = self.label(y)
        
        return logits


class CNN(nn.Module):
    def __init__(self, batch_size, output_size,hidden_size,vocab_size, embedding_length, weights,in_channels=1,out_channels=3,kernel_heights=[1,2,3],stride=1,padding=0,keep_probab=0.1):
        super(CNN, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of each batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
        out_channels : Number of output channels after convolution operation performed on the input matrix
        kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
        keep_probab : Probability of retaining an activation node during dropout operation
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embedding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
        --------

        """
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
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)

        return max_out

    def forward(self, input_sentences, batch_size=None):

        """
        The idea of the Convolutional Neural Netwok for Text Classification is very simple. We perform convolution operation on the embedding matrix 
        whose shape for each batch is (num_seq, embedding_length) with kernel of varying height but constant width which is same as the embedding_length.
        We will be using ReLU activation after the convolution operation and then for each kernel height, we will use max_pool operation on each tensor 
        and will filter all the maximum activation for every channel and then we will concatenate the resulting tensors. This output is then fully connected
        to the output layers consisting two units which basically gives us the logits for both positive and negative classes.

        Parameters
        ----------
        input_sentences: input_sentences of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.
        logits.size() = (batch_size, output_size)

        """

        input = self.word_embeddings(input_sentences)
        # input.size() = (batch_size, num_seq, embedding_length)
        input = input.unsqueeze(1)
        # input.size() = (batch_size, 1, num_seq, embedding_length)
        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)

        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)
        # fc_in.size()) = (batch_size, num_kernels*out_channels)
        logits = self.label(fc_in)

        return logits


##############TRAIN FUNCTIONS##############
def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.Text[0]
        target = batch.Label
        target = torch.autograd.Variable(target).long()
        if (text.size()[0] is not 32):# One of the batch returned by BucketIterator has length different than 32.
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
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
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
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)
    

learning_rate = 2e-5
batch_size = 256
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

''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
# test_sen1 = "You have win a gift, Contact us, our phone number is 888888!!"
# test_sen2 = "hello, could you have time this evening to eat sushi?"
# test_sen1 = TEXT.preprocess(test_sen1)
# test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]
# test_sen2 = TEXT.preprocess(test_sen2)
# test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]
# test_sen = np.asarray(test_sen1)
# test_sen = torch.LongTensor(test_sen)
# test_tensor = Variable(test_sen, volatile=True)
# model.eval()
# output = model(test_tensor, 1)
# out = F.softmax(output, 1)
# if (torch.argmax(out[0]) == 1):
#     print ("it's spam")
# else:
#     print ("it's ham")



################### MAIN FUNCTION ###################

#LODA DATA
TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_dataset()

photo_para = {}
model_list = {}
# model_list['LSTMClassifier'] = LSTMClassifier
# model_list['RCNN'] = RCNN
model_list["CNN"] = CNN

loss_fn = F.cross_entropy
for model in model_list:
    para = experiment(model_list[model],epochs=5)
    photo_para[model] = para

####################TABLE COMPARING DIFFERENT MODELS##############
####################################################################
tab = tt.Texttable() 
headings = ['Names','LOSS','ACCURACY',"TIME(seconds)"]
tab.header(headings) 

fig = plt.figure()
plt.title("the comparazition of precision of different models")
plt.ylabel('Loss')
plt.xlabel('Epoch')


for para in photo_para:
    row = [para,photo_para[para]["test_loss"],photo_para[para]["test_acc"],photo_para[para]["time_needed"]]
    plt.plot(np.arange(len(photo_para[para]["all_train_loss"])), photo_para[para]["all_train_loss"], label=para + str(" train"))
    plt.plot(np.arange(len(photo_para[para]["all_val_loss"])), photo_para[para]["all_val_loss"], label=para + str(" valid"))
    tab.add_row(row)
plt.legend()
plt.show()
s = tab.draw()
print(s)  



###########the comparazition of precision of different models##############
###########################################################################
fig = plt.figure()
plt.title("the comparazition of precision of different models")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

for para in photo_para:
    plt.plot(np.arange(len(photo_para[para]["all_train_acc"])),photo_para[para]["all_train_acc"],label=para)

plt.legend()
plt.show()
