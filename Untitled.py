#!/usr/bin/env python
# coding: utf-8

# In[18]:


#All the necessary imports
import numpy as np
import nltk
import pandas as pd
import os
import re
import csv
import gzip


# In[19]:


directory = "./aclImdb" #Make sure you put the data folder in the same directory as this jupyter notebook file
labeledData = {}
for i in ["train", "test"]:
    labeledData[i] = []
    for sentiment in ["pos", "neg"]:
        score = 1 if sentiment == "pos" else 0
        path = os.path.join(directory, i, sentiment)
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), encoding="utf8") as f:
                labeledData[i].append([f.read(), score])  #Initially adds them to separate lists

np.random.shuffle(labeledData["train"]) #Shuffling
labeledData["train"] = pd.DataFrame(labeledData["train"], columns = ['text', 'sentiment']) #Putting them in a dataframe
np.random.shuffle(labeledData["test"])
labeledData["test"] = pd.DataFrame(labeledData["test"], columns = ['text', 'sentiment'])
labeledData["train"], labeledData["test"] #Prints out both pandas dataframes


# In[20]:


from keras.preprocessing.text import Tokenizer, one_hot, text_to_word_sequence

tokenizer = Tokenizer(num_words=6000)
tokenizer.fit_on_texts(labeledData["train"]["text"])
x_train = tokenizer.texts_to_sequences(labeledData["train"]["text"])
x_test = tokenizer.texts_to_sequences(labeledData["test"]["text"])


# In[21]:


from keras.preprocessing import sequence
max_words = 500
x_train_pad = sequence.pad_sequences(x_train, maxlen=max_words)
x_test_pad = sequence.pad_sequences(x_test, maxlen=max_words)


# In[22]:


tokenizer.word_index
np.amax(x_train_pad) + 1
x_train_pad.shape


# In[23]:


# import gensim
# from keras.preprocessing.text import Tokenizer, one_hot, text_to_word_sequence

# import time

# start = time.time()




# stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
# stop_words = ['in', 'of', 'at', 'a', 'the']
# def tokenize(val):
#     ans = text_to_word_sequence(val)
#     ans = [a for a in ans if not a in stop_words]
#     return ans
# reviews = labeledData["train"]["text"].append(labeledData["test"]["text"], ignore_index=True).values

# words = [tokenize(val) for val in reviews.tolist()]

# model = gensim.models.Word2Vec(sentences=words, size=50, window = 100, workers = 8, min_count=1, sg = 0)


# In[24]:


# model.wv.vocab


# In[25]:


from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, GRU, Bidirectional, GlobalMaxPool1D
vocab_size = np.amax(x_train_pad) + 1
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_words))

model.add(LSTM(16),return_sequences = True)
model.add(LSTM(16))

# model.add(Bidirectional(LSTM(32, return_sequences = True)))
# model.add(GlobalMaxPool1D())
# model.add(Dense(20, activation="relu"))
#model.add(Dropout(0.05))

#model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
print(vocab_size)
model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])

model2 = Sequential()
#model2.add(embedding)


# In[ ]:


model.fit(x_train_pad, labeledData["train"]["sentiment"].values, 
          batch_size = 128, epochs = 1, #validation_split=0.2
         )


# In[ ]:


scores = model.evaluate(x_test_pad, labeledData["test"]["sentiment"], verbose=0)
print('Test accuracy:', scores[1])


# In[ ]:


#1 epoch only
#0.87552 - 128 batch size .2 validation_split using max value 52 seconds
#0.8658 - 128 batch size .2 validation_split using max value 52 seconds
#0.87056 145 s using 5000 32 batch size
#0.85432 196s using max value 32 batch size
#model.

#default: 128 batch size 0 validation_splist 32 lstm size acc: .869
# #default with dropout layer .874
# print(scores)
vocab_size


# In[ ]:


from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test_pad)
acc = accuracy_score(labeledData["test"]["sentiment"], np.around(y_pred))
print("Accuracy score of model: "+str(acc))


# In[ ]:



np.around(y_pred)


# In[ ]:


Embedding(vocab_size, 100, input_length=max_words)


# In[ ]:


models = Sequential()
model.layers[0].get_weights()[0].shape
vocab_size


# In[ ]:


embedding_layer = Embedding([model.layers[0].get_weights()[0]], 100, weights = [model.layers[0].get_weights()[0]], input_length = max_words)


# In[ ]:


import matplotlib.pyplot as plt
def setup_pareto_front(pareto_info, method, func_num, summary_type, gen, trial, auc):
# #     combo = sorted(pareto_info, key=lambda x: x[0])
# #     pop_1 = [a for a, b in combo]
# #     pop_2 = [b for a, b in combo]
    
#     plt.scatter(pop_1, pop_2, color='g')
#     plt.plot(pop_1, pop_2, color='b', drawstyle='steps-post')
# #     plt.xlabel(fitness_index_dict[0])
# #     plt.ylabel(fitness_index_dict[1])
#     plt.title(f"Evolution Type: {method}\nFunction: {func_num}\nBest Pareto Front for gen{gen} trial{trial}\n{summary_type}auc:{auc}") 


# In[ ]:


setup_pareto_front([(327,342), (341,337.25), (342.25,339.5), (337.25, 348.25), (339.75,348.25), (333, 356), (309.75, 380.25), (356, 336.25), ()] , "text classification", "coolness", "idk", 34, 3, 3)


# In[ ]:


# import csv
# with open('movie3.csv', newline='') as lines:
#      spamreader = csv.reader(lines, quotechar='"', delimiter=',',
#                      quoting=csv.QUOTE_ALL, skipinitialspace=True)
#      for row in spamreader:
#         print(row[2:3])
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("movie3.csv", header = None)
df[df.columns[2:4]]




plt.scatter(df[2], df[3], color='g')
# plt.plot(pop_1, pop_2, color='b', drawstyle='steps-post')
df


# In[ ]:


def dominates(a, b):
    """
    Test if a dominates b
    :param a:
    :param b:
    :return: boolean
    """
    return np.all(a <= b)

def my_pareto(data):
    pareto_points = []
    for i, x in enumerate(data):
        pareto_status = True
        pareto_points_to_remove = []
        for j in pareto_points:
            # If a pareto point dominates our point, we already know we are not pareto
            if dominates(data[j], x):
                pareto_status = False
                break
            # If our point dominates a Pareto point, that point is no longer Pareto
            if dominates(x, data[j]):
                pareto_points_to_remove.append(j)
        # Remove elements by value
        for non_pareto_point in pareto_points_to_remove:
            pareto_points.remove(non_pareto_point)
        if pareto_status:
            pareto_points.append(i)
    return pareto_points


# In[ ]:


import numpy as np

list = my_pareto(df[df.columns[2:4]].values)
df = df.iloc[list]
df = df.sort_values(by=[2])
df = df.iloc[9:30-7]
pop1 = df[2]
pop2 = df[3]
plt.scatter(pop1, pop2, s = 20, color='g')
plt.plot(pop1, pop2, color='b', drawstyle='steps-post')


# In[ ]:


from matplotlib.lines import Line2D
st = ["TfidfVectorizer with Single Learner MultiNB", 
"TfidfVectorizer with Single Learner MultiNB", 
      "TfidfVectorizer with Single Learner MultiNB", 
       "TfidfVectorizer with Single Learner MultiNB", 
       "CountVectorizer with Single Learner MultiNB", 
       "CountVectorizer with Single Learner MultiNB", 
       "TfidfVectorizer with Bagged Learner Passive", 
"TfidfVectorizer with Single Learner LinSVC",
 "TfidfVectorizer and Grid Search Learner SGD", 
 "TfidfVectorizer and Single Learner Passive",
 "TfidfVectorizer and Bagged Learner Passive", 
 "Tfidf and Single Learner Passive", 
 "Tfidf and Single Learner Passive", 
      "Tfidf and Single Learner Passive" ]


colors = ['g','g', 'g', 'g', 'lime', 'lime', 'lightcoral', 'y', 'k', 'r', 'lightcoral', 'r', 'r', 'r']
plt.title('Pareto Front for Optimal Individuals in IMDB Dataset')
plt.xlabel("False Positive Rate")
plt.ylabel("False Negative Rate")
falsePos = (df[2]/6249).values
falseNeg = (df[3]/6249).values

for i in range(len(df)):
    stt = st[i]
    pop1 = falsePos[i]
    pop2 = falseNeg[i]
    plt.scatter(pop1, pop2, s = 40, color=colors[i], label = colors[i],)
    #plt.text(pop1+.05, pop2+.05, stt, fontsize=9)
popp1 = falsePos
popp2 = falseNeg
plt.xlim([.030, .06])
plt.plot(popp1, popp2, color='b', drawstyle='steps-post', linewidth = 1)

falseNeg


# In[ ]:


legend_elements = [
                   Line2D([0], [0], marker='o', color='w', label='TfidfVectorizer with Single Learner MultiNB',
                          markerfacecolor='g', markersize=8),
     Line2D([0], [0], marker='o', color='w', label='CountVectorizer with Single Learner MultiNB',
                          markerfacecolor='lime', markersize=8),
     Line2D([0], [0], marker='o', color='w', label='TfidfVectorizer with Bagged Learner Passive',
                          markerfacecolor='lightcoral', markersize=8),
     Line2D([0], [0], marker='o', color='w', label='TfidfVectorizer with Single Learner LinSVC',
                          markerfacecolor='y', markersize=8),
     Line2D([0], [0], marker='o', color='w', label='TfidfVectorizer with Grid Search Learner SGD',
                          markerfacecolor='g', markersize=8),
     Line2D([0], [0], marker='o', color='w', label='TfidfVectorizer with Single Learner Passive',
                          markerfacecolor='r', markersize=8)
]
fig, ax = plt.subplots()
ax.legend(handles=legend_elements, loc = 'center')


# In[ ]:


o = 0
for val in df.iloc[:,1]:
    print(o, val)
    print()
    o+=1

st = ["TfidfVectorizer with Single Learner MultiNB", 
"TfidfVectorizer with Single Learner MultiNB", 
      "TfidfVectorizer with Single Learner MultiNB", 
       "TfidfVectorizer with Single Learner MultiNB", 
       "CountVectorizer with Single Learner MultiNB", 
       "CountVectorizer with Single Learner MultiNB", 
       "TfidfVectorizer with Bagged Learner Passive", 
"Tfidf with Single Learner LinSVC",
 "Tfidf and Grid Search Learner SGD", 
 "Tfidf and Bagged Learner Passive C: 0.1",
 "Tfidf and Single Learner Passive C: 0.01", 
 "Tfidf and Single Learner Passive C: 1 stop_word list: default", 
 "Tfidf and Single Learner Passive C: 1 stop_word list: None"]


# In[ ]:


df2 = pd.read_csv("movie3.csv", header = None)
list = my_pareto(df2[df2.columns[2:4]].values)
df2 = df2.iloc[list]
df2 = df2.sort_values(by=[2])
df2.iloc[24, 1]


# In[ ]:




