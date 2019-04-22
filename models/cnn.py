from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


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

    def name(self):
        return "CNN"
