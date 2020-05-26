#FastText
from fasttext import *
from fasttext.FastText import _FastText

#Data Handling
import pandas as pd
import numpy as np

#Preprocessing
import spacy
import re
from spacy.tokens import Doc
from spacy.lang.en.stop_words import STOP_WORDS
import io
import subprocess
import argparse

from IPython.display import clear_output
from sklearn.metrics import roc_auc_score


class FTEncoder(object):
    """
    Represents an embedding model wrapper based in the fastText learning models to ease workflow loops
    Meant for working on the Clemson Palmetto Cluster but should work anywhere.

    Methods:
        -fit
            -Fits a model to given data.  Supervised or unsupervised depending on whether target outputs are given
        -get_encoding
            -Get a single encoding from a trained model
        -get_all_encodings
            -Get all encodings from a trained model of an entire Series of text instances
        -save_model
            -Saves the underlying fastText model so it can be loaded later
        -load_model
            -Loads a saved fastText model into the underlying fastText model of the object instance

    """
    
    def __init__(self, username = "ndeas", path = "./", load_model = None):
        self.ftModel = _FastText(load_model)
        self.path = path
        self.unsup_training_name = "unsupervised_input.txt"
        self.sup_training_name = "supervised_input.txt"
        self.username = username
        
        
    def fit(self, X, y = None, multilabel = False, preprocess = False, batch_size_pp = 20000, **kwargs):
        """
        Fits the underlying fastText model to the given text input and optional target outputs
        
        Parameters:
            -X
                -A pandas Series of text inputs to train an unsupervised model or supervised model
            -y
                -A pandas Series of labels/target outputs that correspond to the X inputs
            -preprocess
                -Whether the text should be preprocessed or not
            -batch_size_pp
                -The batch size to use if preprocessing.  Large values can crash the kernel, but smaller values take longer
            **kwargs
                -The keyword arguments to pass onto the fastText training method
        """
        if preprocess: self.__class__.__preprocess_text(X, self.username, path = self.path, save_name = self.unsup_training_name, batch_size_pp=batch_size_pp)

        if y is None:
            self.ftModel = train_unsupervised(self.path + self.unsup_training_name, **kwargs)
        else:
            X_pp = self.__class__.__file_to_series(self.path, self.unsup_training_name)
            if multilabel:
                df = pd.concat([X_pp, y], axis = 1)
            else:
                df = pd.DataFrame([X_pp, y])
            self.__class__.__convert_df_to_ft_input(df, df.columns[0], df.columns[1], multilabel, path = self.path, save_name = self.sup_training_name)
            self.ftModel = train_supervised(self.path + self.sup_training_name, loss = 'ova', **kwargs)    
            
            
    def get_params(self, deep = False):
        return {}
    
    def set_params(self, **params):
        return
    
    def predict_proba(self, X):
        return self.ftModel.predict_proba(X)
    
    def get_encoding(self, text):
        """
        Generates a single encoding given a fastText model and text to encode

        Parameters:
            -text
                -The text to encode with ft_model

        Return:
            -The encoding for the sentence when passed to ft_model, in numpy array form
        """
        text = text.replace('\n', ' ')
        enc_vec = self.ftModel.get_sentence_vector(text)
        return enc_vec

    
    def get_all_encodings(self, text_df, text_col):
        """
        Generate encodings for all text instances in a dataframe using the given model

        Parameters:
            -text_df
                -The dataframe holding all input texts to encode
            -text_col
                -The column in text_df that holds the text inputs

        Return:
            -A dataframe identical to text_df, with ane xtra encoding column holding the sentence embeddings
        """

        text_df_w_enc = text_df
        text_df_w_enc["encoding"] = text_df_w_enc[text_col].apply(lambda text: self.get_encoding(text))
        return text_df_w_enc
        
        
    def save_model(self, name):
        """
        Saves the underlying fastText model at the path location
        
        Parameters:
            -name
                -A file name ending in .bin
        """
        
        self.ftModel.save_model(self.path + name)
        
    def load_model(self, name):
        """
        Loads a saved model into the underlying fastText model
        
        Parameters:
            -name
                -A file name ending in .bin
        """
            
        self.ftModel = load_model(self.path + name)
        
    @staticmethod
    def __file_to_series(path, file_name):
        text_list = []
        with open(path + file_name, 'r') as file_list:
#             lines = file_list.read()
#             lines = lines.split('  ')
            
            
            for line in file_list.readlines():
                text_list.append(line)
        return pd.Series(text_list)
        
    @staticmethod
    def __get_en():
        print("getting en")
        cmd = "python3 -m spacy download en_core_web_sm --user"
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        print("Downloaded...")
        output, error = process.communicate()
        if error:
            print("error downloading en for preprocess: {0}".format(error))
        got_en = True

    @staticmethod
    #True flags represent removal of those tokens
    def __preprocess(file, username, punc = True, alpha = True, number = True, Twitter = False, number_words = False, stop_words = True, lemma = True, lower_case = True, email = True, url = True, currency = True, got_en = False):
        
        REPLACE_HASHTAG = re.compile('[#]+[A-Za-z-_]+[A-Za-z0-9-_]')
        REPLACE_AT_ENTITIES = re.compile('[@]+[A-Za-z0-9-_]+')
        REPLACE_HYPHENS = re.compile('(?<=[a-zA-Z])[-][\s]+(?=[a-zA-Z])')

        print("preprocessing ...")
#         if not got_en:
#             print("getting 'en' for spacy")
#             __class__.__get_en()
        nlp = spacy.load('en_core_web_sm')#"/home/" + username + "/.local/lib/python3.6/site-packages/en_core_web_sm/en_core_web_sm-2.1.0")#, disable = ['parser', 'ner']) #TODO: Differences in time and results with these enabled
        
        text = open(file, 'r', encoding='UTF-8').read()

        #Convert hypens from wraparound to a single word. Required for PDF.
        text = REPLACE_HYPHENS.sub('', text)
        #Convert all whitespace to a single space
        text = re.sub(r'[ \t]+',' ', text)

        if Twitter:
            text = REPLACE_HASHTAG.sub('',text)
            text = REPLACE_AT_ENTITIES.sub('',text)

        if lower_case: text = text.lower()

        nlp.max_length = len(text)
        doc = nlp(text)

        tokens = []
        for token in doc:
            add = True
            #Remove email
            if email and token.like_email:
                add = False
            #Remove url
            elif url and token.like_url:
                add = False
            #Remove currency
            elif currency and token.is_currency:
                add = False
            #Remove numbers
            elif number and token.is_digit:
                add = False
            #Remove number words
            elif number_words and token.like_num:
                add = False
            # Remove punctuation
            elif punc and token.is_punct:
                add = False
            #Remove anything containing non characters
            elif alpha and not token.is_alpha:
                add = False
            #Remove stop words
            elif stop_words and token.is_stop:
                add = False
            if add:
                if lemma: tokens.append(token.lemma_)
                else: tokens.append(token.text)
        all_pp_text = ' '.join(tokens)
        return all_pp_text
    
    
    @staticmethod
    def __preprocess_text(data_df, username, path = "./", save_name = "unsupervised_input.txt", batch_size_pp = 20000):
        """
        Preprocess all text in the data frame and return the same dataframe with the preprocessed inputs in a separate column

        Parameters:
            -data_df
                -The dataframe with all text inputs to be used for training
            -text_col
                -The name of the column holding all the text inputs to train the model on
            -path
                -The path where all data and models are saved
            -save_name
                -The name to save the new text input file under.
                *** NOTE: input text file will be saved to path/save_name ***
            -batch_size_pp:
                -The number of text inputs to preprocess simultaneously
                *** Note: Too large batch size can cause the GPU to crash automatically, lower batch_size do not sacrifice much speed ***
        Return:
            -A dataframe identical to data_df except with an extra column ("PP Text") holding the preprocessed text

        """
    
        #Split data for preprocessing ------------------------

        num_splits = (len(data_df)//batch_size_pp) if len(data_df)//batch_size_pp == (1.0*len(data_df)/batch_size_pp) else (len(data_df)//batch_size_pp + 1)
        data_splits = []

        end_token = "endofsentencetoken"

        for split in range(num_splits):
            if(split != num_splits - 1):
                cur_df = data_df[batch_size_pp*split: batch_size_pp*(split + 1)]
            else:
                cur_df = data_df[batch_size_pp*split:]

            data_splits.append(cur_df)


        #Writing training files for preprocessing ------------
        print("Writing training files and preprocessing individually")

        preprocessed_strs = ""

        got_en = False

        for i, data_split in enumerate(data_splits):
            with open(path + "unsup_data_split_" + str(i) + ".txt", 'w', encoding = 'UTF-8') as text_data:
                for text in data_split:
                    text_data.write(str(text) + " " + end_token + "\n ")
                text_data.close()

            #Preprocess each file
            preprocessed_strs += __class__.__preprocess(path + "unsup_data_split_" + str(i) + ".txt", username, got_en = True) + '\n'
            print("Length of preprocessed strings: %d" % len(preprocessed_strs))
            got_en = True
            
            print("Completed preprocessing file " + str(i+1) + " of " + str(len(data_splits)))

        preprocessed_strs_list = preprocessed_strs.split(end_token)[:-1]


        #Compile training files ------------------------------
        print("Recompiling training text files")

        with open(path + save_name, 'w') as outfile:
            for line in preprocessed_strs_list:
                outfile.write(line + '\n')

        print("All text preprocessed, use file name " + save_name + " at the given path for training model")

#         data_df["PP Text"] = preprocessed_strs_list
        return preprocessed_strs_list
    
    @staticmethod
    def __convert_df_to_ft_input(df, text_col, label_col, multilabel = False, path = "./", save_name = "supervised_input.txt"):
        """
        Converts a data frame of labelled text instances to a text file formatted for the fast text supervised model

        Parameters:
            -df
                -The data frame holding the text and labels
            -text_col
                -The name of the column holding the text
            -label_col
                -The name of the column holding the labels
            -path
                -The path where all data should be saved
            -save_name
                -The name not including the path that the training data should be saved under

        """
        
        def __convert_row_to_text(row, text_col, label_col, label_token = "__label__"):
            """
            Convert a single row in the dataframe to just a string using the label token

            Parameters:
                -row
                    -A single row in the dataframe containing the text and labels
                -text_col
                    -The column in the row that holds the text of the instance
                -label_col
                    -The column in the row that holds the label of the instance
                -label_token
                    -the token that should be used to distinguish the text from the label

            Return:
                -A single string holding the text + label_token + label
            """
            text_in = str(row[text_col])
            if multilabel:
                labels = row[label_col]
                try:
                    for label in labels:
                        if not len(label) < 2:
                            text_in += label_token + str(label) + ' '
                except:
                    pass
            else:
                text_in +=  label_token + str(row[label_col])
            return text_in #str(row[text_col]) + label_token + str(row[label_col])
    
        label_token = "__label__"
        converted_series = df.apply(lambda row: __convert_row_to_text(row, text_col, label_col, label_token), axis = 1)

        with open(path + save_name, 'w') as sup_input:
            for instance in converted_series:
                if len(instance.replace('__label__', '')) > 2:
                    sup_input.write(str(instance).replace('\n', '') + "\n")
            sup_input.close() 