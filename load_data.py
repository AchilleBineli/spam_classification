import pandas as pd
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
from numpy.random import RandomState

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
