import threading

import spacy
import os

import nltk
from nltk.stem.porter import *
import string
import math as m
import numpy as np
from tkinter import*
import pandas as pd
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import ScreenManager,Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.base import runTouchApp
from kivy.clock import Clock
class SearchEngine:
    docs = "Corpus"  # **folder that contains the set of documents
    nlp = spacy.load("en_core_web_lg")  # **use spacy library as it supports natural language processing
    i=''
    Term_Freq = {}  # create dictionary to contain each term with his termfreq in each doc
    Doc_ID = 1
    Positional_index = {}
    Score_details = {}
    N = 0
    Sum_tf_idf = 0
    doc_dict={}
    doc_vector = {}
    Files_dict={}
    doc_tf_idf= {}
    doc_length ={}
    def __init__(self):
        pass

    def get_filename( self, path):
        return [self.i.path for self.i in os.scandir(path) if self.i.is_file()]
    def File_list(self):
        files =self.get_filename(self.docs)
        return files

    # ##################################################
    def remove_punctation(self ,tokens):
        table = str.maketrans('', '', string.punctuation)  # support translate method as it convert each char to none
        stripped = [w.translate(table) for w in tokens]  # used to remove punctations
        return stripped

    ####################################################
    def convert_tokens_to_lowerCase(self ,tokens):
        token_list = [word.lower() for word in tokens]
        return token_list

    ####################################################
    def stem_tokens(self ,tokens):
        stemmer = PorterStemmer()
        stem_result = [stemmer.stem(tok) for tok in tokens]
        return stem_result

    #####################################################
    def lemmatization_tokens(self ,tokens):
        sp_tokens = self.nlp(str(tokens))  # in order to see it as a token not string type
        lemm_tokens = [tok.lemma_ for tok in sp_tokens]
        return lemm_tokens
    #################################################
    def Positional_index_model(self ,list_of_tokens):

        i = 1
        for position, term in enumerate(list_of_tokens,start=1):
            # enumerate adds a counter (position of term) convert list to list of tuples [(0,'team')] and so on

            # check if that term is already exist in our model or not
            if term in self.Positional_index:

                # check if that term in the same doc_id or not
                if self.Doc_ID in self.Positional_index[term][1]:
                    # true therefore will add the new position in the list
                    self.Positional_index[term][1][self.Doc_ID].append(position)

                    self.Term_Freq[term][self.Doc_ID] = len(self.Positional_index[term][1][self.Doc_ID])
                    # as term freq is the length of positions

                else:
                    # in case that term already exist in our model but at another doc_id we will add the new dictionary for them
                    self.Positional_index[term][1][self.Doc_ID] = [position]
                    self.Positional_index[term][0] = self.Positional_index[term][0] + 1  # doc freq

                    self.Term_Freq[term][self.Doc_ID] = len(self.Positional_index[term][1][self.Doc_ID])
            # if that term does not exist in our model then we will add all details
            else:
                # first we add our term as a key then the value of key is list
                # that list will contain doc_freq and doc_id and all positions of that term
                self.Positional_index[term] = []
                self.Term_Freq[term] = {}  # we will use it in tf-idf
                self.Term_Freq[term][self.Doc_ID] = 1
                # first we will append docfreq in list
                self.Positional_index[term].append(1)  # we add here one as it is the first encounter so it always will be one
                self.Positional_index[term].append({})  # as a second element in list we add dictionary
                # that dictionary contains doc_id as a key and list of position as a value
                self.Positional_index[term][1][self.Doc_ID] = [position]


    ########################################################
    # after preprocessing documents we also call positional index model
    def preprocess_docs(self):

        # iterate for each file to do preprocessing and call our model
        files =self.get_filename(self.docs)
        for filepath in files:
            name, extension = os.path.splitext(filepath)
            if extension == '.docx':
                pass
                #convert_docx_to_txt(filepath)
            elif extension == '.txt':
                with open(filepath, 'r', encoding="utf8", errors="surrogateescape") as file_to_read:
                    some_text = file_to_read.read()
                print(os.path.basename(filepath))
                result = [tok.text for tok in self.nlp.tokenizer(some_text) if not tok.is_stop]
                tokens_after_remove_pun = self.remove_punctation(result)

                lowercase_tokens = self.convert_tokens_to_lowerCase(tokens_after_remove_pun)
                lemma_tokens = self.lemmatization_tokens(lowercase_tokens)
                stemmer_tokens = self.stem_tokens(lemma_tokens)
                final_out = self.remove_punctation(stemmer_tokens)  # repeated it to remove again all punctation
                while 'n' in final_out: final_out.remove('n')

                str_list = list(filter(None, final_out))  # that to remove the empty strings from list
                #print(str_list)
                #print('-' * 40)1
                self.doc_dict[self.Doc_ID]=str_list
                self.Files_dict[self.Doc_ID]=filepath
                self.Positional_index_model(str_list)
                self.Doc_ID = self.Doc_ID + 1
                print('-' * 40)
    ###########################################
    ##################################################
    # tf for documents
    # score details = {'term':{'doc_id':  doc id  ,'tf': tf , 'tfw': tfw} , idf , tf-idf  , unit vector]}} {'term':{docid: [tf , tfw]}}
    def tf_weighted(self):
        termfreq = self.Term_Freq
        for term, info in termfreq.items():
            self.Score_details[term] = {}
            for Doc_id in info:
                if info[Doc_id] > 0:
                    Tfw = 1 + m.log10(info[Doc_id])
                    self.Score_details[term][Doc_id] = []
                    self.Score_details[term][Doc_id].append(info[Doc_id])
                    self.Score_details[term][Doc_id].append(Tfw)


        return self.Score_details
    #####################################################

    def idf_weighted(self):
        self.N = len( self.get_filename(self.docs))
        position_index = self.Positional_index
        score_details = self.tf_weighted()
        for term in position_index:
            for doc_id in score_details[term]:
                idf = m.log10((self.N / position_index[term][0]))
                score_details[term][doc_id].append(idf)
        return score_details
    ######################################################
    def tf_idf(self):
        score_details = self.idf_weighted()
        for term in score_details:
            for doc_id in score_details[term]:
                tf_idf = (score_details[term][doc_id][1] * score_details[term][doc_id][2])
                score_details[term][doc_id].append(tf_idf)

        for doc_id in self.doc_dict:
            for i in range(len(self.doc_dict[doc_id])):
                if self.doc_dict[doc_id][i] in score_details:
                    if doc_id in self.doc_tf_idf:
                        self.doc_tf_idf[doc_id].append(score_details[self.doc_dict[doc_id][i] ][doc_id][3])

                    else:
                       self.doc_tf_idf[doc_id] = [score_details[self.doc_dict[doc_id][i] ][doc_id][3]]

        return score_details
    ##############################################################
    def calculate_doc_length(self):

        for doc_id in self.doc_tf_idf:
            sum = 0
            for i in range(len(self.doc_tf_idf[doc_id])):
                sq= m.pow(self.doc_tf_idf[doc_id][i],2)
                sum=sum+sq
            self.doc_length[doc_id] =m.sqrt(sum)
        return self.doc_length
    def unit_vector(self):
        score_details = self.tf_idf()
        length_doc = self.calculate_doc_length() # length doc is a dictionary that contains docid with its length
        print('score details inside unit vector function')
        print(score_details)
        for term in score_details:
            for doc_id in score_details[term]:
                if length_doc[doc_id] ==0.0:
                    unit_vector=0.0
                    score_details[term][doc_id].append(unit_vector)
                    if doc_id in self.doc_vector:
                        self.doc_vector[doc_id].append(unit_vector)

                    else:
                        self.doc_vector[doc_id]=[unit_vector]


                else:
                    unit_vector = (score_details[term][doc_id][3] / length_doc[doc_id])
                    score_details[term][doc_id].append(unit_vector)
                    if doc_id in self.doc_vector:
                        self.doc_vector[doc_id].append(unit_vector)

                    else:
                        self.doc_vector[doc_id] = [unit_vector]
        print('score details inside unit vector function before return')
        print(score_details)
        return score_details
    ###################################################################
    def format_positional(self):
        for term in self.Positional_index:
            print("<" + str(term) + ":" + str(self.Positional_index[term][0]) + ";")
            for doc_id in self.Positional_index[term][1]:
                print(str(doc_id) + ":", end='')
                for pos in self.Positional_index[term][1][doc_id]:
                    print(str(pos) + ",", end='')
                print(';')
            print('>')
    ###################################################################
    def df_format(self):
        terms_list = list(self.Positional_index.keys())
        doc_freq_list = []
        # doc_ids =[]
        # positions = []
        Doc_ids_Positions = []
        for term in terms_list:
            doc_freq_list.append(self.Positional_index[term][0])
            #doc_ids.append(list(Positional_index[term][1].keys()))
            #positions.append(list(Positional_index[term][1].values()))
            Doc_ids_Positions.append(self.Positional_index[term][1])


        data = {'Term': terms_list, 'Doc_Freq': doc_freq_list, ' Doc_Ids : [Positions] ': Doc_ids_Positions}
        df = pd.DataFrame(data, columns=['Term', 'Doc_Freq', ' Doc_Ids : [Positions] '])
        print(df)
class Query:
    document_class = SearchEngine()
    pos_intersection = {}
    Key_ID = 0
    store_exist_terms = {}  # that dictionary will contain all existing terms in both positional index and query
    Term_Query_freq ={}
    Query_score_details = {}
    Sum_tf_idf_guery=0
    query_vector =[]
    def __init__(self):
        pass
    # preprocessing query
    def preprocess_query(self,query):
        result = [tok.text for tok in self.document_class.nlp.tokenizer(query) if not tok.is_stop]
        tokens_after_remove_pun = self.document_class.remove_punctation(result)

        lowercase_tokens = self.document_class.convert_tokens_to_lowerCase(tokens_after_remove_pun)
        lemma_tokens = self.document_class.lemmatization_tokens(lowercase_tokens)
        stemmer_tokens = self.document_class.stem_tokens(lemma_tokens)
        final_out = self.document_class.remove_punctation(stemmer_tokens)  # repeated it to remove again all punctation
        str_list = list(filter(None, final_out))  # that to remove the empty strings from list
        print(str_list)
        print('-' * 40)
        return str_list
    ##########################################################################
    # Matching query
    def Dict_keyID_similar_terms(self, query):

        list_of_Prepros_query = self.preprocess_query(query)
        print(list_of_Prepros_query)
        for term in list_of_Prepros_query:
            if term in self.document_class.Positional_index:

                if term not in self.store_exist_terms:
                    # store_exist_terms[term]= Positional_index[term]
                    # store_exist_terms[term].append()
                    self.store_exist_terms[self.Key_ID] = []
                    self.store_exist_terms[self.Key_ID].append(term)
                    self.store_exist_terms[self.Key_ID].append(self.document_class.Positional_index[term])
                    self.Key_ID = self.Key_ID + 1

        return self.store_exist_terms

    ##############################################################################
    def position_intersect( self, store1, store2, doc_ids_list):

        for doc_id in doc_ids_list:
            i = 0
            j = 0

            pos_list1 = store1[doc_id]
            pos_list2 = store2[doc_id]

            while (i < len(pos_list1) and j < len(pos_list2)):
                if (pos_list2[j] - pos_list1[i]) == 1:
                    self.pos_intersection[doc_id] = []
                    self.pos_intersection[doc_id].append(pos_list2[j])
                    i += 1
                    j += 1
                else:
                    if pos_list1[i] < pos_list2[j]:
                        i += 1
                    else:
                        j += 1
        return self.pos_intersection
    ##################################################################
    def intersect_docs( self, store1, store2):

        intersection_docs_list = list(store1.keys() & store2.keys())
        final_result = self.position_intersect(store1, store2, intersection_docs_list)

        return final_result
    ##################################################################
    def Matching_Query(self,query):
        j = -1
        result = []
        store_terms = self.Dict_keyID_similar_terms(query)
        for i in range(len(store_terms)):

            if i == 0:
                store1 = store_terms[i][1][1]
                store2 = store_terms[i + 1][1][1]
                result.append(self.intersect_docs(store1, store2))
                j += 1
            elif i == 1:
                continue
            else:
                store1 = result[j]
                store2 = store_terms[i][1][1]
                result.append(self.intersect_docs(store1, store2))
                j += 1
        return result[j - 1]
    ###################################################################
    def duplicate_terms_in_query(self, Query):
        query = self.preprocess_query(Query)
        for term in query:
            if term in self.Term_Query_freq:
                self.Term_Query_freq[term] += 1
            else:
                self.Term_Query_freq[term] = 1

        return self.Term_Query_freq
    #######################################################################

    def tf_idf_query(self , query):
        term_Query_freq = self.duplicate_terms_in_query(query)
        score_details = self.document_class.Score_details
        for term in term_Query_freq:
            if term_Query_freq[term] > 0:
                Tfw = 1 + m.log10(term_Query_freq[term])
                self.Query_score_details[term] = []
                self.Query_score_details[term].append(term_Query_freq[term])
                self.Query_score_details[term].append(Tfw)
            if term in score_details:
                for doc_id in score_details[term]:
                    idf_query = score_details[term][doc_id][2]
                self.Query_score_details[term].append(idf_query)
            tf_idf = (self.Query_score_details[term][1] * self.Query_score_details[term][2])
            self.Query_score_details[term].append(tf_idf)
        return self.Query_score_details
    #####################################################################

    def unit_vector_query(self, query):

        query_score_details = self.tf_idf_query(query)
        for term in query_score_details:
            square_tf_idf = (query_score_details[term][3] * query_score_details[term][3])
            self.Sum_tf_idf_guery = self.Sum_tf_idf_guery + square_tf_idf
        length_tf_idf = m.sqrt(self.Sum_tf_idf_guery)
        for term in query_score_details:
            if length_tf_idf == 0:
                unit_vector = 0.0
                query_score_details[term].append(unit_vector)
                self.query_vector.append(unit_vector)
            else:
                unit_vector = (query_score_details[term][3] / length_tf_idf)
                query_score_details[term].append(unit_vector)
                self.query_vector.append(unit_vector)

        return query_score_details
###########################################################################
class score:
    cos_similarity = {}
    doc_class = SearchEngine()
    query_class = Query()
    prod_term ={}
    prod_cos_similar =0
    dict_score={}
    doc_product={}
    Score_doc ={}
    def __init__(self):
        pass
        #
        # for doc_id in dic_doc_vector:
        #     new_dic_doc_vector = list(dic_doc_vector[doc_id])
        #     new_list_query_vector = list(list_query_vector)
        #
        #     max_dim = [np.max(np.array(dic_doc_vector[doc_id]).shape), np.max(list_query_vector.shape)]
        #     addition_num = abs(max_dim[1] - max_dim[0])
        #
        #     if np.array(dic_doc_vector[doc_id]).shape > list_query_vector.shape:
        #         temp = list(np.zeros(int(addition_num)))
        #         for e in temp:
        #             new_list_query_vector.append(e)
        #
        #         #print('new_q_vector  ' + str(new_list_query_vector))
        #         #print('new_dic_doc_vector  ' + str(new_dic_doc_vector))
        #     elif np.array(dic_doc_vector[doc_id]).shape < list_query_vector.shape:
        #
        #         temp = list(np.zeros(int(addition_num)))
        #         for e in temp:
        #             new_dic_doc_vector.append(e)

    def cosine_similarity(self):
        score_details = self.doc_class.unit_vector()
        print('unit vector doc')
        print(score_details)
        for term in self.query_class.Query_score_details:
            if term in score_details:
                for doc_id in score_details[term]:
                    product_uV = (self.query_class.Query_score_details[term][4]*score_details[term][doc_id][4])
                    score_details[term][doc_id].append(product_uV)
                    if doc_id in self.doc_product:
                        self.doc_product[doc_id].append(product_uV)

                    else:
                        self.doc_product[doc_id] = [product_uV]
        #print(score_details)
        for doc_id in self.doc_product:
            self.Score_doc[doc_id]=sum(self.doc_product[doc_id])




     # that to find score of cosine similarity by sum all cosine similarity of terms for each document
        return self.Score_doc
    ####################################################
    def Sort_Scoring(self):
        s = self.Score_doc
        sorted_score = sorted(s.items(), key = lambda kv:(kv[1], kv[0]) , reverse=True)
        print(sorted_score)
        r=[str(self.doc_class.Files_dict[doc_id]) for doc_id , score in sorted_score]
        sorted_score={}
        return r

    def format_score(self):
        doc_id_wit_tf ={}
        doc_id_wit_tfw = {}
        doc_id_wit_idf = {}
        doc_id_wit_tf_idf = {}

        doc_id_wit_uV = {}
        for term in self.doc_class.Score_details :
            for doc_id in self.doc_class.Score_details[term]:
                doc_id_wit_tf[doc_id]=[]
                doc_id_wit_tf[doc_id].append(self.doc_class.Score_details[term][doc_id][0])
                doc_id_wit_tfw[doc_id] = []
                doc_id_wit_tfw[doc_id].append(self.doc_class.Score_details[term][doc_id][1])
                doc_id_wit_idf[doc_id] = []
                doc_id_wit_idf[doc_id].append(self.doc_class.Score_details[term][doc_id][2])
                doc_id_wit_tf_idf[doc_id] = []
                doc_id_wit_tf_idf[doc_id].append(self.doc_class.Score_details[term][doc_id][3])


    # s = SearchEngine()
    # q = Query()
    # ss = score()
    # def pressed(self):
    #
    #     n = self.s.preprocess_docs()
    #     # print(s.Positional_index)
    #
    #     y = self.s.tf_weighted()
    #     k = self.s.idf_weighted()
    #     uu = self.q.unit_vector_query(self.query.text)
    #
    #     print(self.ss.cosine_similarity())
    #     print(self.ss.Sort_Scoring())
       # self.query.text = ""
####################################################################################################

class GUI(Widget): #kv file ,the name of that file should be all small cases that file like css file for design
    query = ObjectProperty(None)
    btn = ObjectProperty()
    result_str = ""
    s = SearchEngine()
    q = Query()
    ss = score()

    def __init__(self, **kwargs):
        super(GUI, self).__init__(**kwargs)
    def pressed(self):
        self.process(self.query.text)
        print(self.lst)
        if len(self.lst)==1:
              # ,size_hint=(None, None),height=30,width=180,halign="center"
            self.res =TextInput(multiline=True,text=str(self.lst),size_hint=(None, None),height=100,width=320,halign="center")
            self.add_widget(self.res)
            self.query.text=""

            Clock.schedule_once(self.callback, 10)

            self.lst=''



            #self.query.text = str(lst)
        elif len(self.lst)>1:
            set_lst =list(set(self.lst))
            for i in range(len(set_lst)):
                self.result_str =self.result_str + str(set_lst[i]) + " , \n"
                self.res =TextInput(multiline=True,text=str(self.result_str), size_hint=(None, None), height=100, width=320,
                                      halign="center")
                self.add_widget(self.res)
                self.query.text = ""
            Clock.schedule_once(self.callback, 10)

            self.lst=''


    def callback(self, dt):
        print('In callback')
        self.res.text = ''
        self.res.readonly = True
        self.res.text = ' In order to search again, \n you need to stop and rerun ^-^ \n notice "scroll down and up to see result "'
        self.lst=''




    def process(self,_query):
        n = self.s.preprocess_docs()
        # print(s.Positional_index)

        y = self.s.tf_weighted()
        k = self.s.idf_weighted()
        uu = self.q.unit_vector_query(_query)

        print(self.ss.cosine_similarity())
        print(self.ss.Sort_Scoring())
        self.lst = self.ss.Sort_Scoring()
        print(len(self.lst))













class Window_manager(ScreenManager):
    pass
    # def __init__(self,**kwargs):
    #     super(GUI,self).__init__(**kwargs) # **kwargs this is stands that we can pass as many as argument we want
    #     self.cols=1
    #     self.inside = GridLayout()
    #     self.inside.rows = 1
    #     self.inside.add_widget(Label(text="[b]Search in your text documents![/b] \n",font_size=18,markup=True))
    #     self.add_widget(self.inside)
    #     self.again= GridLayout()
    #     self.again.cols=1
    #     self.again.rows=2
    #     self.query = TextInput(multiline=False) #,size_hint=(None, None),height=30,width=180,halign="center"
    #     self.again.add_widget(self.query)
    #     self.button =Button(text=" [b]Google it [/b] ",markup=True)
    #     self.again.add_widget(self.button)
    #     self.button.bind(on_press=self.pressed)
    #     self.add_widget(self.again)


class Search_EngineAPP(App): # create a kivy interface and build an app by inheriting from app class

    def build(self): # that's for building your app you have to return a widget on the build() function
        return GUI()






# s = SearchEngine()
#
# n =s.preprocess_docs()
# #print(s.Positional_index)
# q=Query()
# Query_= input('Enter your Query : \n ')
# #print(q.Matching_Query(Query_))
# #s.df_format()
# y =s.tf_weighted()
# k =s.idf_weighted()
#
#
#
#
#
# #print(s.doc_v_d)
# uu= q.unit_vector_query(Query_)
# #print(s.doc_tf_idf)
# #print(s.calculate_doc_length())
# # print('query details ')
# # print(q.Query_score_details)
# # print('doc details ')
# # print(s.Score_details)
#
# ss= score()
# print(ss.cosine_similarity())
# print(ss.Sort_Scoring())
Search_EngineAPP().run()

#s.format_positional()
#print(ss.dict_score)
#print(ss.Sort_Scoring())
 # it is similar to create an object from the class

#Classroom creativity also corresponds
# qq= q.preprocess_query('car insurance')
# print(qq)
# print(s.Positional_index)
# print(q.Matching_Query('car insurance'))
# print(q.store_exist_terms)
# print(q.unit_vector_query(' insurance best  insurance car insurance car'))
# Score = score()
# print(s.Term_Freq)