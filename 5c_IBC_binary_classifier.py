'''
@author: mahfuz

Prepare balanced dataset for IBC (using adfontes)
Build two independent models and compare them
    - USE-based model
    - Naive bayes model
'''

import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from random import sample
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
import re
import pickle
from nltk.tokenize import word_tokenize
import preprocessing # Use Adriens' way of preprocessing

SEED = 42 # random seed for reproducibility


'''
scp rahma118@mcclintock.cs.umn.edu:/project/csbio/Mahfuz/Analysis_and_Papers/0_Playground/Omdena/Task5/Task5c_political_bias/My_Try/Political_bias_10000.csv .
scp politics_no_processing.tsv rahma118@mcclintock.cs.umn.edu:/project/csbio/Mahfuz/Analysis_and_Papers/0_Playground/Omdena/Task5/Task5c_political_bias/My_Try/
'''

# =============================================================================
#                      1. Read and balance IBC data 
# =============================================================================
df_IBC = pd.read_csv("IBC_data_2_labels.csv") # (4326, 2)
df_IBC.Label[df_IBC.Label == 'biased'] = 1
df_IBC.Label[df_IBC.Label == 'not'] = 0
df_IBC.Label = df_IBC.Label.astype(int)
df_IBC.Label.value_counts() # 3726 (biased), 600 (not biased)

'''
Balance IBC data using adfontes data
Add 1000 normal / not biased sentences from adfontes
'''
# Article To Sentence
import unicodedata
import unidecode
from spacy.lang.en import English
nlp = English()
nlp.add_pipe('sentencizer')

def paragraph_to_sentences(para):
    doc = nlp(unidecode.unidecode(unicodedata.normalize("NFKD", para)))
    sentences = [sent.text.strip(", ").strip("\", ").strip(" \"") for sent in doc.sents]
    return sentences

df_adfontes = pd.read_csv('adfontes_political_bias.csv') # sum(df_adfontes.article_bias == 0) # 486
df_adfontes = df_adfontes.loc[df_adfontes.article_bias == 0, :, ]
df_adfontes = df_adfontes.reset_index()

all_sentences = []
for i in range(len(df_adfontes)):
	sentences_text = paragraph_to_sentences(df_adfontes.article_text[i])
	for text in sentences_text:
		all_sentences.append(re.sub(r'\n', '', text))

all_sentences = [re.sub(' +', ' ', x) for x in all_sentences] # Remove consecutive spaces
all_sentences_valid = [x for x in all_sentences if ((len(x) > 10) & (len(x) < 500))] # 14466
new_df = pd.DataFrame({'Sentence': sample(all_sentences_valid, 3000), 'Label': [0] * 3000}) 
df_IBC = df_IBC.append(new_df) # (7326, 2)
df_IBC.to_csv('IBC_2labels_balanced_using_adfontes.csv', index = False)
'''
df_IBC.Label.value_counts()
1    3726
0    3600
'''


# =============================================================================
#                      2. Pre-processing of data
# =============================================================================
df_IBC = pd.read_csv("IBC_2labels_balanced_using_adfontes.csv") # (7326, 2)
df_IBC.Sentence = df_IBC.Sentence.astype(str)

# ------------------------ Way1: Adrien's pre-processing ------------------------
df_IBC["Sentence"] = df_IBC.apply(lambda row: preprocessing.preprocess(row['Sentence']), axis = 1) # 7326 * 2


# ------------------------ Way2: Simple pre-processing ------------------------
# https://github.com/Keerthiraj-Nagaraj/NLP-ML-for-US-political-data-analysis/tree/master/Data%20visualization%20%2B%20Machine%20learning%20for%20political%20data%20analysis
'''
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
def clean_text(text):
    """
        text: a string        
        return: modified initial string
    """
#     text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

df_IBC["Sentence"] = df_IBC['Sentence'].apply(clean_text)
'''

# Way 3: Should we use googles preprocessing
# For BERT: https://github.com/google-research/bert/blob/master/tokenization.py

# Way 4: No pre-processing
# # https://www.kaggle.com/xhlulu/disaster-nlp-train-a-universal-sentence-encoder


# =============================================================================
#                         3. Modeling
# =============================================================================
x_train, x_test, y_train, y_test = train_test_split(df_IBC['Sentence'], 
                                                    df_IBC['Label'], 
                                                    test_size=0.2, 
                                                    stratify=df_IBC['Label'], 
                                                    random_state=SEED)


# ------------ Model1a: Adrien's pre-processing Use USE + sigmoid (dense) layer ------------
model = tf.keras.models.Sequential()
model.add(hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4', 
                        input_shape=[], 
							dtype=tf.string, 
                        trainable=True))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)
history = model.fit(x_train, 
          y_train, 
          epochs=10,
          callbacks=[callback],
          validation_split = 0.1)
# len(history.history['loss'])

# Evaluation
y_pred_test = model.predict(x_test)
y_pred_test[y_pred_test <= 0.5] = 0
y_pred_test[y_pred_test > 0.5] = 1

print(classification_report(y_test, y_pred_test))
print(confusion_matrix(y_test, y_pred_test))
'''
              precision    recall  f1-score   support
           0       0.92      0.88      0.90       720
           1       0.89      0.93      0.91       746

    accuracy                           0.90      1466
   macro avg       0.90      0.90      0.90      1466
weighted avg       0.90      0.90      0.90      1466

[[631  89]
 [ 55 691]]
'''


# ------------ Model1b: Simple pre-processing + USE + sigmoid (dense) layer ------------

''' Quite Similar to Adrien's pre-processing
              precision    recall  f1-score   support
           0       0.91      0.88      0.90       720
           1       0.89      0.92      0.90       746

    accuracy                           0.90      1466
   macro avg       0.90      0.90      0.90      1466
weighted avg       0.90      0.90      0.90      1466

[[633  87]
 [ 59 687]]
'''


# ------------ Model1c: No pre-processing + USE + sigmoid (dense) layer ------------
''' Quite Similar to Adrien's pre-processing
              precision    recall  f1-score   support
           0       0.88      0.92      0.90       720
           1       0.92      0.88      0.90       746

    accuracy                           0.90      1466
   macro avg       0.90      0.90      0.90      1466
weighted avg       0.90      0.90      0.90      1466

[[663  57]
 [ 87 659]]
'''


# Make some figures
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# https://neptune.ai/blog/keras-metrics


# ------------ Model 4: Naive Bayes + simple pre-processing (baseline) ------------
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

model_nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
model_nb.fit(x_train, y_train)
y_pred_test = model_nb.predict(x_test)
print('accuracy %s' % accuracy_score(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))
print(confusion_matrix(y_test, y_pred_test))

'''
              precision    recall  f1-score   support

           0       0.96      0.75      0.84       720
           1       0.80      0.97      0.88       746

    accuracy                           0.86      1466
   macro avg       0.88      0.86      0.86      1466
weighted avg       0.88      0.86      0.86      1466

[[542 178]
 [ 23 723]]
'''

# y_pred_test = model_nb.predict_proba(x_test) # For probability scores (required for curves)



# =============================================================================
#                         4. Comparison (USE vs NB)
# =============================================================================
''' ---------------- PR curve and ROC curve ----------------
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html#sphx-glr-auto-examples-miscellaneous-plot-display-object-visualization-py
https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
'''

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

# 1. USE
y_pred_test = model.predict(x_test)
prec, recall, _ = precision_recall_curve(y_test, y_pred_test, pos_label=1)
fpr, tpr, _ = roc_curve(y_test, y_pred_test, pos_label=1)

# 2. NB
y_pred_test = model_nb.predict_proba(x_test)
prec_nb, recall_nb, _ = precision_recall_curve(y_test, y_pred_test[:,1], pos_label=1)
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_pred_test[:,1], pos_label=1)


# Plot side by side
lw = 2

# ROC curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Political bias: IBC (balanced)', fontsize=16)

ax1.set_title('ROC Curve')
ax1.set(xlabel='False Positive Rate', ylabel='True Positive Rate')

ax1.plot(fpr, tpr, lw = lw, color='#6baed6', label = 'USE-based')
ax1.plot(fpr_nb, tpr_nb, lw = lw, color='#74c476', label = 'Naive Bayes')
ax1.plot([0, 1], [0, 1], color='#969696', lw=lw, linestyle='--', label = 'Random')
ax1.legend(frameon=False, loc='lower center', ncol=2)

# PR curve
ax2.plot(recall, prec, lw = lw, color='#6baed6', label = 'USE-based')
ax2.plot(recall_nb, prec_nb, lw = lw, color='#74c476', label = 'Naive Bayes')
ax2.set_title('PR Curve')
ax2.set(xlabel='Recall', ylabel='Precision')
# plt.show()
plt.savefig('IBC_binary_USE_NB_Performance.png', dpi = 100)
