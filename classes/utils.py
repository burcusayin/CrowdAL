#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:31:56 2019

@author: burcusyn
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import random

class Vectorizer():
    ''' The class that transforms text data to vectors. '''
    def __init__(self):
        self.vectorizer = TfidfVectorizer(min_df=3, max_features=None, 
            strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
            stop_words = None, lowercase=False)

    def transform(self, X):
        return self.vectorizer.transform(X).toarray()

    def fit(self, X):
        self.vectorizer.fit(X)

    def fit_transform(self, X):
        return self.vectorizer.fit_transform(X).toarray()
    
class VoteAggretagor():
    ''' The class that includes methods for vote aggregation. '''
    def majorityVoting(self, voteHistory, datapoint):
        votes = voteHistory[datapoint]
        vote_count = Counter(votes)
        vote_count_values = list(vote_count.values())
        if len(vote_count_values) > 1 and (vote_count_values[0] == vote_count_values[1]):
            majorityVote = random.sample([0,1], 1)[0]
        else:
            majorityVote = int(vote_count.most_common(1)[0][0])
        return majorityVote