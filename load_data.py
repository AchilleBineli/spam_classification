import pandas as pd
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
from numpy.random import RandomState

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import seaborn as sns; sns.set()
import tensorflow as tf

def load_dataset(test_sen=None): 
    
    def _load_data_torch_text():
        tokenize = lambda x: x.split()
        TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
        LABEL = data.LabelField()
        train_data,test_data = data.TabularDataset.splits(
            skip_header=True,
            path='./data/', train ='train.csv',test='test.csv' ,format='csv',
            fields=[('Label', LABEL),('Text', TEXT)])
        TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
        LABEL.build_vocab(train_data)
        word_embeddings = TEXT.vocab.vectors
        print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
        print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
        print ("Label Length: " + str(len(LABEL.vocab)))
        train_data, valid_data = train_data.split() 
        train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32, sort_key=lambda x: len(x.Text), repeat=False, shuffle=True)
        vocab_size = len(TEXT.vocab)
        TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = _load_data_torch_text()
 
    
    def _load_data_pd():
        data = pd.read_csv("./data/spam.csv")
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
        # number of words used as features
        max_features = 10000
        # cut off the words after seeing 500 words in each document(email)
        maxlen = 500



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
        # shuffle data
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

        texts_train = data[:training_samples]
        y_train = labels[:training_samples]
        texts_test = data[training_samples:]
        y_test = labels[training_samples:]
        
        return texts_train,y_train,texts_test,y_test

    data_origin = {}
    TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = _load_data_torch_text()
    data_origin["from_torch_text"] = [TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter]
    texts_train,y_train,texts_test,y_test = _load_data_pd()
    data_origin["from_pd"] = [texts_train,y_train,texts_test,y_test]
    
    return data_origin