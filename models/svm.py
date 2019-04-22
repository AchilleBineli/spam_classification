import numpy as np
from sklearn import svm
from keras.layers import  Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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