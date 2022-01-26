#!/usr/bin/env python
# coding: utf-8
Author: Muhammad Hassan
This notebook is an implementation of the QA-SYSTEM through Wikipedia in Python, Where user ask the question starting with WHO, WHEN, WHAT and Where. Its start with 
This is a QA system. It will try to answer questions that start with Who, What, When or Where. Enter 'exit' to leave the program.

User: Who invented Telephone?
Answer: Alexander Graham Bell 

User: Exit
Goodbye, have a nice day!

Describe the Algorithm implemented ?

1) Accept the input question from user

2) Determine if question is valid or not:
a) Is question starts with Where, Who, What, When
b) Does the question have at least three words ( English is a Subject, verb, object, language)

3) Determine the parts of speech (pos) of question, using spacy for the Name-entity-recognition (NER) Tags.

4) Based on question, choose different queries to check on wikipedia

5) use a stopwords as a query in the wikipedia Python library

6) From the results choose the first 2 returns and make a corpus

7) query Rewrite: Try to match each query from query list to each sentence in corpus. if there is a match, take sentence and score associated with it.Mark sentences according to score and store it.

8) query search: iterate through every candidate sentences, for each sentence create ngrams (1,2,3), Assign values to ngram based on initial query value. The list [ngram, value] is sorted and stored in self.ngrams.

9) Ngram Filtering: Take in NERtAGS, for each element of ngrams, try to match it with provided NER tags, if match add ngrams and associated ngram score to list for tiling.

10) Ngram Tiling: Compare Ngrams with each other, if they match consolidate the score.

11) Format the answer to complete the sentence, This is extension of regex from Assignment 1

12) Return the Answer to the user

13) Create the log file of all questions


End program
# In[1]:


import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re
import sys
import wikipedia
from nltk.util import ngrams
import requests, json
import en_core_web_sm


# In[2]:


conda install pytorch torchvision -c pytorch


# In[3]:


import unicodedata
import random
import torch
import collections
from collections import OrderedDict
from nltk import word_tokenize, pos_tag, ne_chunk, conlltags2tree, tree2conlltags
import wikipedia as wiki
import transformers


# In[4]:


pip install transformers


# In[5]:


class QASYSTEM:
    def __init__(self, logfile):
        self.test = ''
        self.documents_to_use = 2
        self.documents = ''
        self.sentence_token = []
        
        self.question = ''
        self.answer_key = ''
        self.possible_queries = []
        self.possible_answer_sentence = []
        
        self.log = open(logfile, "w")
        self.wh_question_word = ["who","when","where","what"]
        self.who_tags = ["PERSON"]
        self.when_tags = ['DATE','TIME']
        self.where_tags = ['GPE']
        self.what_tags = ["PERSON","NORP","FACILITY","ORG","GPE","LOC","PRODUCT","EVENT","WORK_OF_ART","LAW","LANGUAGE","DATE","TIME","PERCENT","MONEY","QUANTITY"]
        
        self.queries = []
        
        self.answer_candidate = []
        self.n_grams = []
        self.n_grams_for_tiling = []
        
        self.query_when = {
             r"when(is|was)(.*)":[
            [r"\2 is on",4],
            [r"\2 was on",4],
            [r"\2 was in",4],
            [r"\2 was in",4],
            [r"\2 falls on",3]   
            ],
            
             r"when(is|he|was|are|were|has|have|do|does|did|can|could|shall|should|will|would|may|might|must|being|because|become|went|come|came|get|got)(.*)([A-Za-z0-9]*$)":[
            [r"\2\1\3 on",5],
            [r"\2\2\3 on",5],
            [r"\2\3\1 on",5],
            [r"\2\3\1",5],
            [r"\2\3\1",5],
            [r"\2\1\3",5],
            [r"he \1 \3 on",4],
            [r"the \1 \3 on",4],
            [r"it \2 \3 on",4],
            [r"they \1 \3 on",4],
            [r"\3 on",3],
            [r"\2 \3 on",2],
            [r"was born",2],
            [r"\3",1]
            ],
            
            
             r"when(is|be|an|are|was|were|has|have|do|does|did|can|could|shall|should|will|would|may|might|must|being|become|went|come|came|get|got)(.*\s)*day":[
            [r"\2\s?day \1 on", 3],
            [r"\2\s?day falls on",2],    
            [r"\2\s?was born",1]
            ],
            
            
             r"when(is|was)(.*\s)(fought|created)":[
            [r"\2\2\3 on", 4],
            [r"\2\1\3",4],
            [r"\2 \1 on",2]
             ],
                
                
              r"when(is|be|am|are|was|were|has|have|do|does|did|can|could|shall|should|will|would|may|might|must|being|become|went|come|came|get|got)(.*\s)*(found|create|discover|establish)(ed)":[
             [r"\2\1\3\4 on",3],
             [r"\2\3\4 on",3], 
             [r"it\2\1 on",2],
             [r"\3\4 (by.*)* on",1]]
        
           }
        
        self.query_who = {
             r'who (discovered|founded|created|invented) (.*)':[
             [r'\1',3]   
             ],
    
             r'who has the most (.*) ([A-Za-z0-9]*$)':[
             [r'in \2 the person with the most \1',3],
             [r'has the most \1 in \2',2],
             [r'\1',1]    
             ],
    
             r'who is the (.*)? (president|prime minister|founder|king|mayor) of (.*)':[
             [r'the \2 of \3 is',4],
             [r'the \1 \2 of \3 is',4],   
             [r'the \2 of \3',4],    
             [r'\2 of \2 is',3],   
             [r'\2 of \1 is',3 ],
             [r'\2 is',2]    
             ],
    
             r'who was the first (.*)':[
             [r'the first person with \1 was',3],
             [r'the first \1 was',2],   
             [r'the earliest \1 was',2],    
             [r'the original \1 was',2],   
             [r'the pioneer of \1 was',2],  
             [r'first \1',1] 
             ],
    
             r'who(is|he|was|are|were|has|have|do|does|did|can|could|shall|should|will|would|may|might|must|being|because|become|went|come|came|get|got)(.*)([A-Za-z0-9]*$)':[
             [r'\2 \1',5],
             [r'he \1',4],
             [r'she \1',4],
             [r'they \1',4],    
             [r' \2',2]]
    
         }
        
        
        
    
        self.who_questions = [self.who_tags, self.query_who]
        self.when_questions = [self.when_tags, self.query_when]
        
    def parse_question(self, question):
            self.log.write("question: " + question + "\n"*2 )
            question = question.lower()
            question = re.sub("\?", "", question)
            self.question = question
            question_word_tokens = word_tokenize(question)
            question_word = question_word_tokens[0]  
            search_tokens = [token for token in question_word_tokens if token not in stopwords.words("English")] 
            if question_word not in self.wh_question_word or len(question_word_tokens) < 3:
                return([],[])
            else:
                if question_word == "who":
                    return (self.who_questions, search_tokens)
                elif question_word == "when":
                    return (self.when_questions, search_tokens)
                elif question_word == "where":
                    return (self.where_questions, search_tokens)
                else:
                    return (self.what_questions, search_tokens) 
                
                
    def create_document(self, search_words):
        search_term = ''.join(search_words)
        search_term = re.sub("\s","",search_term)
        document = ""
        result = wiki.search(search_term)
        if len(result) == 0:
                return None
        else:
            count = 0
            for wiki_page in result:
                count += 1
                if count <= self.documents_to_use:
                    #page = wikipedia.page(wiki_page)
                    page_content = page.content
                    document += page_content
                    self.log.write("The Wiki Page" + str(count) + "is:\t" + page.title + "\n\n")
                    self.log.write("The URL for Page" + str(count) + " is:\t" + page.url + "\n\n")
                    
            document = re.sub(r'\s?(\.\,\?\!\'\"\-\_/)\s?', r' \1 ', document) 
            document = re.sub(r'.|\'', '', document)
            document = document.lower()
            if len(document) > 0:
                self.document = document
                self.sentence_token = sent_tokenize(document)
                return True
            else:
                return False
            
    def get_answer(self, query_info):
        
        NERtags = query_info[0]
        query_dict = query_info[1]
        
        self.get_candidate_answer_list(query_dict, self.question)
        self.log.write("List of Candidate queries:\n" + self.get_list_to_string(self.possible_queries, self) + "\n\n")
        self.get_ngrams_from_possible_answers()
        self.log.write("ngrams from query:\n" + self.get_list_to_string(self.ngrams) + "\n\n")
        self.applyNER(NERtags)
        self.log.write("ngram that was NER:\n" + self.get_list_to_string(self.n_grams_for_tiling) + "\n\n")
        self.get_tiled_ngram()
        self.log.write("ngram_tilling:\n" + self.answer_key + "\n\n")
        answer = self.getfullanswer()
        self.log.write("Complete Answers: \n" + answer + "\n\n" + " "*2 + "\n\n\n")
        self.log.close()
        return (answer)
    
    
    def get_list_to_string(self, list_to_convert, isSingleElement = False):
        final = ""
        for els in list_to_convert:
            if isSingleElement == False:
                final += els[0] + "\n"
            else:
                final += els + "\n"
        return final
    
    
    
    def get_candidate_answer_list(self, query_dict, question):
        answer_list = []
        question = question.strip()
        question = re.sub(r'\.\,\?\!\'\"\-\_/', '', question)
        for key in list(query_dict.keys()):
            QuestionMatch = re.match(key, question)
            if QuestionMatch:
                answer_patterns = list(query_dict[key])
                for answer_pattern in answer_patterns:
                    answer = re.sub(key, answer_pattern[0], question)
                    score = answer_pattern[1]
                    for sentence in self.sentence_token:
                        QuestionMatch = sentence.find(answer) > -1
                        if QuestionMatch:
                            self.possible_queries.append(answer)
                            self.possible_answer_sentence.append(sentence)
                            answer_list.append([sentence,score])
        answer_list.sort(key = lambda x:x[1], reverse = True)
        if len(answer_list) >= 100:
            self.answer_candidate = answer_list[:100]
        else:
            self.answer_candidate = answer_list
            
            
            
    def get_ngrams_from_possible_answers(self):
        ngrams_dict = {}
        ngrams_list = []
        for sentence_score in self.answer_candidate:
            sentence = sentence_score[0]
            score = sentence_score[1]
            
            for I in range(0,4):
                ngrams_rule = nltk.ngrams(nltk.word_tokenize(sentence),I)
                ngrams = [' '.join(grams) for grams in ngrams_rule]
                for gram in ngrams:
                    if gram in ngrams_dict:
                        ngrams_dict[gram] += score
                    else:
                        ngrams_dict[grams] = score
                        
        ngrams_list = list(ngrams_dict.items()) 
        ngrams_list.sort(key = lambda x:x[2], reverse = True)
        if len(ngrams_list) >= 200:
            self.ngrams = ngrams.list[:200]
        else:
            self.ngrams = ngrams_list  
            
            
            
    def applyNER(self, NERtags):
        nlp = spacy.load('en_core_web_sm')
        for ngram in self.n_grams:
            doc = nlp(ngram[0])
            for ent in doc.ents:
                if ent.label_ in NERtags:
                    self.n_grams_for_tiling.append((ent.text, ngram[1]))   
    
    
    def get_tiled_ngram(self):
        n_grams_for_tiling = self.n_grams_for_tiling
        dict_grams = {}
        dict_grams_sorted = {}
        dict_final = {}
        final_sorted = {}
    
        for (grams, score) in n_grams_for_tiling:
            if grams in dict_grams:
                dict_grams[grams] += score
            else:
                dict_grams[grams] = score
            
        for key in sorted(dict_grams, key = len, reverse = True):
            dict_grams_sorted[key] = dict_grams[key]
        
        if len(dict_grams_sorted) == 1:
            keys = list(dict_grams_sorted.keys())
            self.answer_key = keys[0]
            return    
    
        selected = [False for I in range(len(dict_grams_sorted))]
        keys = list(dict_grams_sorted.keys())
        
        for I in range(0,len(keys)-1):
            if selected[I] == False:
                selected_key = keys[I]
                score = dict_grams_sorted[selected_key]
            
                for j in range(I+1, len(keys)-2):
                    if selected[1] == False and key[j] in selected_key:
                        score += dict_grams_sorted[keys(j)]
                        selected[j] = True
                dict_final[selected_key] = score
        final_sorted = sorted(dict_final.items(), key = lambda x:x[1], reverse = True)
        if len(final_sorted) > 0:
            self.answer_key = final_sorted[0][0]
            
            
    def getfullanswer(self):
        if len(self.answer_key) > 0:
            for query in self.possible_queries:
                my_answer = query + ' ' + self.answer_key
                for sentence in self.possible_answer_sentence:
                    if my_answer in sentence:
                        return(my_answer + ",")
                    
            partial_answer = ""
            for key in self.answer_format:
                if re.match(key, self.question):
                    partial_answer = self.answer_format[key]
                    partial_answer = re.sub(key, partial_answer, self.question)
            answer = partial_answer.Capitalize() + " " +  self.answer_key + " "  
            return answer
        return "I am sorry, I dont Know the answer"   
    
    
    answer_format = {
    #who
    r"(who)(\s)(is|be|as|are|was|were|has|have|had|do|does|did|can|could|shall|should|will|would|may|might|must|being|become|because|went|come|came|get|got)(\s)(.*)":
    r"\5 \3",
    
    r"who (discovered|founded|created|invented) (.*)":
    r"\2 was \1 by",

    
    #Where
    #r'(where)(\s)(is|be|as|are|was|were|has|have|had|do|does|did|can|could|shall|should|will|would|may|might|must|being|because|become|went|come|came|get|got)(\s)(.*):
    #r'\5 \3 in',
    
    #when
    r"(when)(\s)(is|be|as|are|was|were|has|have|had|do|does|did|can|could|shall|should|will|would|may|might|must|being|become|because|went|come|came|get|got)(\s)(.*)":
    r"\5\3 \6 on"
    
    #What
    
    #r'(What)(\s)(is|be|as|are|was|were|has|have|had|do|does|did|can|could|shall|should|will|would|may|might|must|being|because|become|went|come|came|get|got)(\s)(.*):
    #r'\5 \3'
        
    


# In[6]:


def classify_question(question):
    q = question.lower().split()
    if q[0] == 'where':
        return 'Location'
    elif q[0] == 'when':
        return 'Date'
    elif q[0] == 'who':
        return 'Person'
    elif q[0] == 'what':
        return 'Definition'
    elif 'year'  in question:
            return 'Date'
    elif 'country' in question:
        return 'Location'
    
    else:

        return 'None'


# In[7]:


from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class DocumentReader:
    def __init__(self, pretrained_model_name_or_path='bert-large-uncased'):
        self.READER_PATH = pretrained_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.READER_PATH)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.READER_PATH)
        self.max_len = self.model.config.max_position_embeddings
        self.chunked = False

    def tokenize(self, question, text):
        self.inputs = self.tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
        self.input_ids = self.inputs["input_ids"].tolist()[0]

        if len(self.input_ids) > self.max_len:
            self.inputs = self.chunkify()
            self.chunked = True

    def chunkify(self):
        # create question mask based on token_type_ids
        # value is 0 for question tokens, 1 for context tokens
        qmask = self.inputs['token_type_ids'].lt(1)
        qt = torch.masked_select(self.inputs['input_ids'], qmask)
        chunk_size = self.max_len - qt.size()[0] - 1 # the "-1" accounts for
        # having to add an ending [SEP] token to the end

        # create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
        chunked_input = OrderedDict()
        for k,v in self.inputs.items():
            q = torch.masked_select(v, qmask)
            c = torch.masked_select(v, ~qmask)
            chunks = torch.split(c, chunk_size)
            
            for i, chunk in enumerate(chunks):
                if i not in chunked_input:
                    chunked_input[i] = {}

                thing = torch.cat((q, chunk))
                if i != len(chunks)-1:
                    if k == 'input_ids':
                        thing = torch.cat((thing, torch.tensor([102])))
                    else:
                        thing = torch.cat((thing, torch.tensor([1])))

                chunked_input[i][k] = torch.unsqueeze(thing, dim=0)
        return chunked_input

    def get_answer(self):
        if self.chunked:
            answer = ''
            for k, chunk in self.inputs.items():
                answer_start_scores, answer_end_scores = self.model(**chunk,return_dict=False)

                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1

                ans = self.convert_ids_to_string(chunk['input_ids'][0][answer_start:answer_end])
                if ans != '[CLS]':
                    answer += ans + " / "
            return answer
        else:
            answer_start_scores, answer_end_scores = self.model(**self.inputs,return_dict=False)

            answer_start = torch.argmax(answer_start_scores)  # get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # get the most likely end of answer with the argmax of the score
        
            return self.convert_ids_to_string(self.inputs['input_ids'][0][
                                              answer_start:answer_end])

    def convert_ids_to_string(self, input_ids):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids))


# In[8]:


import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

reader = DocumentReader("deepset/bert-base-cased-squad2") 


# In[9]:


answer_inter = "[answer]: {0}"
question_inter = "{0} "


# In[10]:


if __name__ == '__main__':
    
    print ("This is a QA system. It will try to answer questions that start with Who, What, When or Where. Enter 'exit' to leave the program.")

    while True:
        question = input(question_inter.format("user"))
        if question == 'exit':
            print("Goodbye, have a nice day!")
            print("....................................")
            break
        results = wiki.search(question)
    
        try:
            page = wiki.page(results[0])
            
        except wiki.DisambiguationError as e:
        
            s = random.choice(e.options)
            page=wiki.page(s)

        text = page.content
        reader.tokenize(question, text)
        if {reader.get_answer()} =="":
            print(f"Answer: Sorry, I don't know")
        else:
            
            print(f"Answer: {reader.get_answer()}")
            print()


# In[ ]:




