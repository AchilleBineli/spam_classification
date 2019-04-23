import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from string import punctuation
import numpy as np
import pandas as pd

def data_visualization():
    # Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
    data = pd.read_csv('./data/smsspamcollection/SMSSpamCollection',
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