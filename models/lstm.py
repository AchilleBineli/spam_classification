import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

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