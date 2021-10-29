#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DataScienceHW 
@File    ：node2vec.py
@IDE     ：PyCharm 
@Author  ：Yaser
@Date    ：2021/10/29 21:27 
@Describe: 
"""
from scipy.io import loadmat
import networkx as nx
import numpy as np
import json
import random
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from gensim.models import KeyedVectors


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        problabel = np.zeros((probs.shape[0], probs.shape[1]))
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            problabel[i, probs_.argsort()[-k:]] = 1
        return problabel


def get_prob(pre, cur):
    unnormalized_probs = []
    for cur_nbr in sorted(G.neighbors(cur)):
        if cur_nbr == pre:
            unnormalized_probs.append(G[cur][cur_nbr]['weight'] / p)
        elif G.has_edge(cur_nbr, pre):
            unnormalized_probs.append(G[cur][cur_nbr]['weight'])
        else:
            unnormalized_probs.append(G[cur][cur_nbr]['weight'] / q)
    norm_const = sum(unnormalized_probs)
    normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
    return np.array(normalized_probs)


def sample(array):
    return np.random.choice(len(array), p=array)


if __name__ == '__main__':

    adjmatrix = loadmat('file/blogcatalog.mat')['network']
    G = nx.from_numpy_matrix(adjmatrix.todense())
    nodes = list(G.nodes())
    directed = 0
    p = 0.25
    q = 0.25
    num_walks = 10
    walk_length = 40
    node_transition = {}
    for node in nodes:
        unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        node_transition[node] = np.array(normalized_probs)
    edge_transition = {}

    if directed:
        for edge in G.edges():
            edge_transition[edge] = get_prob(edge[0], edge[1])
    else:
        for edge in G.edges():
            edge_transition[edge] = get_prob(edge[0], edge[1])
        edge_transition[(edge[1], edge[0])] = get_prob(edge[1], edge[0])
    walks = []
    for walk_iter in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    next = cur_nbrs[sample(node_transition[cur])]
                else:
                    prev = walk[-2]
                    next = cur_nbrs[sample(edge_transition[(prev, cur)])]
                walk.append(next)
            else:
                break
            walks.append(walk)

    d = 128
    k = 10
    from gensim.models import Word2Vec

    model = Word2Vec(walks, vector_size=d, window=k, min_count=0,workers=25)
    model.wv.save_word2vec_format('deepwalk_blogcatalog.txt')
    model = KeyedVectors.load_word2vec_format('deepwalk_blogcatalog.txt', binary=False)
    labels_matrix = loadmat('file/blogcatalog.mat')['group']
    features_matrix = np.asarray([model[str(node)] for node in range(labels_matrix.shape[0])])
    training_percents = 0.9
    X, y = skshuffle(features_matrix, labels_matrix)
    training_size = int(training_percents * X.shape[0])
    X_train = X[:training_size]
    y_train = y[:training_size]
    X_test = X[training_size:]
    y_test = y[training_size:]
    clf = TopKRanker(LogisticRegression())
    clf.fit(X_train, y_train)
    top_k_list = np.diff(y_test.tocsr().indptr)
    preds = clf.predict(X_test, top_k_list)
    result = f1_score(y_test, preds, average='macro')
