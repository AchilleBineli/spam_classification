import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

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
    
    def name(self):
        return "RCNN"