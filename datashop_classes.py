
import pandas as pd
import numpy as np
import random

import math

import matplotlib.pyplot as plt
import textdistance
import matplotlib as mpl

import nltk
from nltk import word_tokenize
from nltk.stem.porter import *
nltk.download('punkt')

from sklearn.feature_extraction import text as sk_text
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.feature_extraction import text
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

stemmer = PorterStemmer()

def word_counts(
    df_main,
    col, 
    stopwords='english',
    tokenizer = None,
    show_chart = True, 
    vocab = None,
    cap = 30):

    countvec = CountVectorizer(
        stop_words=stopwords,
        tokenizer = tokenizer,
        vocabulary = vocab
        )
    stemmer = PorterStemmer()

    df_wordcounts= countvec.fit_transform(df_main[col])

    words = countvec.get_feature_names()
    sums = df_wordcounts.toarray().sum(axis=0)
    count_dict = dict(zip(words,sums))

    count_series = pd.Series(count_dict)
    count_series.sort_values(ascending = False,inplace=True)

    if show_chart == True:
        plot_wordcounts(count_series)

    return count_series

def plot_wordcounts(series,cap = None):

    if not cap:
        cap = 30
    
    top_30 = series.head(cap)

    fig,ax = plt.subplots(figsize=(30,20))

    ax.tick_params(
        axis='x', 
        labelrotation=35,
        length=10,
        labelsize= 26
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(False)

    ax.bar(list(top_30.index),top_30)

    fig.tight_layout()

    fig.show()

def compare_vocabs(seta,setb):
    left_words = set(seta.index)
    right_words = set(setb.index)

    left_notin_right = list(left_words - right_words)
    sr_right_notin_left = seta[left_notin_right]
    sr_right_notin_left.sort_values(ascending=False,inplace=True)
    plot_wordcounts(sr_right_notin_left)

def wordcount_matrix(df, text_col
    stop_words='english',
    tokenizer=None,
    vocabulary=None,
    ngram_range=ngram_range)

class NLP_Factory:
    ''' Fits and evaluates LDA or NMF topic model. 

        Streamlines Topic Modeling Workflow by vectorizing the corpus,
        fitting a model and assessing how closely calculated components
        align with existing labels. 

        Arguments:
        dataframe -- Holding the text and any labels. Data will be appended
        text_col -- (str) name of the column containing text corpus
        labels_col -- (str) name of column with the labels to comapre to topics
        model_type -- (str) 'lda' or 'nmf'
        n_topics -- (int) the number of model components to create
        stop_words -- (str or list) the stop words to exclude
        

        Methods:
        Upon initialization, the first three methods are called in
        reverse sequence. These are the core methods of the class. 

        fit_mod -- instantiates either a CountVectorizer object and 
            a LDA object, or a TfidfVectorizer object and an NMF object
            based on model_type value passed in at instantiation, with 
            n_topics number of components. The text is vectorized and 
            passed into the model. Component numebrs for each document
            are added back to the dataframe. 

        mod_reporting -- Calculates the distribution of original labels
            in the set of documents that are assigned each given model
            component number. 
            Also gathers the highest occuring label for the component,
            as long as this was not the highest occuring label for a 
            previously evaluated component, into a dictionary. This
            is used in the next method to assigne a label to a component. 

        map_topics --
            assigns a label to a component based on the dictionary 
            caculated from the previous method. Also calculates a column
            that checks if the original label matches the label assigned
            by the model and the mapping. 

        dist_charts -- 
            returns a figure with subplots each showing the distribution
            of original labels for a given component

        acc_report -- 
            calculates the rate at which the original label matches the 
            label assigned by the map_topics method for each label, as
            well as the overall match rate. '''

    rand_state = 42
    import math
    
 
    def __init__(
        self,
        dataframe,
        text_col,
        labels_col = None,
        tokenizer = None,
        ngram_range = (1,1),
        vocabulary = None,
        model_type = 'lda' ,
        n_topics =7, 
        desparsify = False,
        stop_words= 'english',
        description = None,
        topic_matrix_addback = False
        ):
        
 
        self.data = dataframe
        #self.n_topics = n_topics
        #self.stop_words = stop_words
        self.text_col = text_col
        self.desparsify = desparsify
        self.topic_matrix_addback = topic_matrix_addback
        self.labels_col = labels_col
        self.model_type = model_type

        self.vect_params = {
            'stop_words':stop_words,
            'tokenizer':tokenizer,
            'vocabulary':vocabulary,
            'ngram_range':ngram_range
        }
        
        self.topic_params = {
            'n_components':n_topics,
        }

        self.fit_mod()

        if self.labels_col:
            self.map_topics()
            self.report_row = {
                'Description':description,
                'Model Type':self.model_type,
                'Stop Words':self.vect_params['stop_words'],
                'Accuracy':self.acc_score,
            }

    
    def fit_mod(self):
 
        if self.model_type == 'lda':
            self.vect_object = CountVectorizer(**self.vect_params)
            self.mod_object = LatentDirichletAllocation(**self.topic_params)
        elif self.model_type == 'nmf':
            self.vect_object = TfidfVectorizer(**self.vect_params)
            self.mod_object = NMF(**self.topic_params)
        else:
            raise Exception('must specify model type as lda or nmf')
 
        self.vectors = self.vect_object.fit_transform(self.data[self.text_col])
        

        self.mod_object.fit(self.vectors)
    
        self.topic_matrix = self.mod_object.transform(self.vectors)
        self.data['mod_number'] = self.topic_matrix.argmax(axis=1)

        if self.desparsify == True:
            self.vector_df = pd.DataFrame(self.vectors.todense(),columns=self.vect_object.get_feature_names())
            self.data = pd.concat([self.data,self.vector_df],axis=1)
        
        if self.topic_matrix_addback == True:
            self.topic_df = pd.DataFrame(self.topic_matrix)
            self.topic_df.columns = ['t_' + str(topic) for topic in self.topic_df.columns]
            self.data = pd.concat([self.data,self.topic_df],axis=1)
 
    def mod_reporting(self):
        
        self.mod_nums = list(self.data['mod_number'].value_counts().index)
 
        self.mod_series = {}
        self.topic_map = {}
 
        for numb in self.mod_nums:
            # Slice all rows with this mod topic number. Get original label colum
            self.sr_labels = self.data[self.data['mod_number'] == int(numb)][self.labels_col]
 
            # Assign value counts of orignal labels for this  
            self.mod_series[numb] = self.sr_labels.value_counts()
            self.labels_ordered = list(self.sr_labels.value_counts().index)
                                
            i = 0
            while self.labels_ordered[i] in self.topic_map.values():
                i += 1
                if i >= len(self.labels_ordered):
                    break
            else:
                self.topic_map[numb] = self.labels_ordered[i]
 
        self.mod_report = pd.concat(self.mod_series,axis=1).fillna(0)
 
    def map_topics(self):
        self.mod_reporting()
        self.data.loc[:,'mod_label'] = self.data['mod_number'].replace(self.topic_map)
        self.data.loc[:,'mod_match'] = self.data[self.labels_col] == self.data['mod_label']
        self.acc_score = sum(self.data.loc[:,'mod_match'])/len(self.data)
 
    def dist_charts(self):
        self.chart_num = len(self.mod_nums)
 
        if self.chart_num % 5 == 0:
            self.row_num = int(self.chart_num/5)
        else:
            self.row_num = int(math.ceil(self.chart_num/5))
 
        self.lda_fig,self.ax = plt.subplots(
            nrows=self.row_num,
            ncols=5,
            figsize = (40,7*self.row_num))
        
        self.lda_fig.canvas.draw()
 
        for col,ax in zip(list(self.mod_report.columns),self.ax.flatten()):
            ax.bar(
                self.mod_report.index,
                self.mod_report[col]
            )
 
            ax.tick_params(
                axis='x', 
                labelsize=12,
                labelrotation=45,
                length=0
                )
            
            ax.set_title(
                label='LDA Number: ' + str(col),
                loc='right',fontdict={
                    'color':'#434343',
                    'fontsize':18,
                    },
                    y=0.7
                    )
 
            self.lda_fig.canvas.draw()
            self.xlabels_new = [label.get_text().replace(' ', '\n') for label in ax.get_xticklabels()]
            ax.set_xticklabels(self.xlabels_new)
 
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
 
        self.lda_fig.tight_layout()
        self.lda_fig.show()
 
    def acc_report(self):
 
            self.acc_dict = {}
            self.df_groupby = self.data.groupby(self.labels_col)
 
            for name,group in self.df_groupby:
 
                self.labels = group[self.labels_col]
                self.predicts = group['mod_label']
 
                self.acc_dict[name]= sum(self.labels == self.predicts)/len(self.labels)
                
            
            self.acc_series = pd.Series(self.acc_dict)
            self.acc_series['OVERALL'] = self.acc_score
            self.acc_series.sort_values(inplace=True)
            self.acc_series.plot(figsize=(12,12),kind='barh',xlim= (0,1))



'__________________________________________________________________________________________________'

class Refinery():
    '''Fits and keeps track of the performance of multiple pipelines '''

    def __init__(self,X,y):
        self.run_id = 1
        self.models={}
        self.report = pd.DataFrame(
            columns=['description','f1_score','accuracy']
            )
        
        self.load_data(X,y)
    
    def load_data(self,X,y):
        self.X = X
        self.y = y

    def ingest(self,pipeline,info = None, show_report= False):
        self.info = info
        self.pipeline = pipeline
        self.models[self.run_id] = Batch(
                                    self.X,
                                    self.y,
                                    pipeline,
                                    info)
        
        self.report = self.report.append(
            self.models[self.run_id].row_dict,
            ignore_index=True)

        self.run_id += 1

        display(self.report)

        
class Batch():
    '''Runs and evaluates performance of given pipeline '''

    def __init__(self,X,y,pipeline,info = None):

        self.X = X
        self.y = y
        self.pipeline = pipeline
        self.info =info 

        self.scoring = {
            'accuracy' : make_scorer(
                metrics.accuracy_score),  
            'f1_score' : make_scorer(
                metrics.f1_score, average = 'weighted')}


        self.class_dict = cross_validate(
            self.pipeline,self.X,self.y,scoring=self.scoring,cv=6)
        
        self.row_dict ={
            'Model':type(self.pipeline[0]).__name__,
            #'Vectorizer':type(self.pipeline[0]).__name__,
            'accuracy':self.class_dict['test_accuracy'].mean(),
            'f1_score':self.class_dict['test_f1_score'].mean()
        }

        if self.info:
            self.row_dict.update(self.info)