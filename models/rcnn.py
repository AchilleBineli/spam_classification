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
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) 
        self.dropout = 0.8
        self.lstm = nn.LSTM(embedding_length, hidden_size, dropout=self.dropout, bidirectional=True)
        self.W2 = nn.Linear(2*hidden_size+embedding_length, hidden_size)
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

    def name(self):
        return "RCNN"