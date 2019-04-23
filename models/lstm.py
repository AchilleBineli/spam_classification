
from keras.layers import SimpleRNN, Embedding, Dense, LSTM
from keras.models import Sequential
import time
import numpy as np



def LSTMClassifer(data,epochs=3):
    epochs = epochs
    texts_train,y_train,texts_test,y_test = data
    debut = time.time()
    lstm_para={}
    # texts_train,y_train,texts_test,y_test
    model = Sequential()
    model.add(Embedding(10000, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history_ltsm = model.fit(texts_train, y_train, epochs=epochs, batch_size=60, validation_split=0.2)
    
    pred = model.predict_classes(texts_test)
    acc = model.evaluate(texts_test, y_test)
    proba_ltsm = model.predict_proba(texts_test)
    fini = time.time()
    lstm_para["all_train_acc"] = (np.array(history_ltsm.history['acc'])*100).tolist()
    lstm_para["all_val_acc"] = (np.array(history_ltsm.history['val_acc'])*100).tolist()
    lstm_para["all_train_loss"] = history_ltsm.history['loss']
    lstm_para["all_val_loss"] = history_ltsm.history['val_loss']
    lstm_para["test_loss"] = acc[0]
    lstm_para["test_acc"] = acc[1]*100
    lstm_para["time_needed"] = fini - debut
    return lstm_para
