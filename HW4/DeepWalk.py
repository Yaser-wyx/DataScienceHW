#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DataScienceHW 
@File    ：DeepWalk.py
@IDE     ：PyCharm 
@Author  ：Yaser
@Date    ：2021/10/29 16:09 
@Describe: 
"""

import numpy as np
import networkx as nx
import argparse
import warnings
import matplotlib.pyplot as plt
from time import perf_counter
from datetime import timedelta
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from multiprocessing import cpu_count
import random
import scipy.io
from itertools import repeat
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score


def build_corpus(G, max_paths, path_len):
    corpus = build_walk_corpus(G, max_paths, path_len)
    print("Number of randowm walks in the corpus = ", len(corpus))
    return corpus


################################################################################
def generate_embeddings(d, w, hs, corpus):
    print("Word2Vec parameters: Dimensions = " + str(d) + ", window = " + str(w) + ", hs = " + str(
        hs) + ", number of cpu cores assigned for training = " + str(cpu_count()))

    model = Word2Vec(vector_size=d, epochs=4, window=w, sg=1, min_count=0, hs=hs, compute_loss=True,
                     workers=cpu_count())
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    word_vec = model.wv
    return word_vec


################################################################################
def eval_classifier(G, subs_coo, word_vec):
    # F1 score function.
    results = evaluate(G, subs_coo, word_vec)
    for i in results.keys():
        print("--> ", i)

    trainsize = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_results = {}
    for (name, res) in results.items():
        print("Classifier: ", name)
        for (tr_size, res_) in zip(trainsize, res):
            print("Training size : ", tr_size)
            print("Micro F1: ", res_[0])
            print("Macro F1: ", res_[1])

        avg = np.average(res, axis=0)
        print("---------------------------------------")
        print("Average Micro F1 : ", avg[0])
        print("Average Macro F1 : ", avg[1])
        plot_graph(trainsize, res)


################################################################################
def process(args):
    dimensions = args.d
    max_paths = args.walks
    path_len = args.len
    window = args.window
    hs = args.hs

    G, subs_coo = parse_mat_file(args.file)
    corpus = build_corpus(G, max_paths=args.walks, path_len=args.len)
    word_vec = generate_embeddings(dimensions, window, hs, corpus)
    # Evaluate the embeddings by passing it through classifier(s)
    eval_classifier(G, subs_coo, word_vec)


################################################################################
def parse_mat_file(path):
    edges = []
    G = nx.Graph()
    mat = scipy.io.loadmat(path)
    nodes = mat['network'].tolil()
    subs_coo = mat['group'].tocoo()
    for start_node, end_nodes in enumerate(nodes.rows, start=0):
        for end_node in end_nodes:
            edges.append((start_node, end_node))
    G.add_edges_from(edges)
    G.name = path
    print(nx.info(G))
    return G, subs_coo


################################################################################
def random_walk(G, start_node, path_len):
    path = [str(start_node)]
    current = start_node
    while len(path) < path_len:
        neighbors = list(G.neighbors(current))
        if len(neighbors) == 0:
            break
        current = random.choice(neighbors)
        path.append(str(current))
    return path


################################################################################
def remove_self_loops(G):
    loops = []
    # Get loops
    for i, j in G.edges_iter():
        if i == j:
            loops.append((i, j))
    G.remove_edges_from(loops)
    return G


################################################################################
def build_walk_corpus(G, max_paths, path_len):
    corpus = []
    nodes = list(G)
    # all iterations
    for path_count in range(max_paths):
        random.Random(0).shuffle(nodes)
        corpus = corpus + list(map(random_walk, repeat(G), nodes, repeat(path_len)))
    print("all iterations for random walk Completed")
    return corpus


################################################################################
def sparse2array_inv_binarize(y):
    # range(y.shape[1])
    mlb = MultiLabelBinarizer()
    mlb.fit(y)
    y_ = mlb.inverse_transform(y.toarray())
    return y_


################################################################################
def custom_predict(classifier, X_test, y_test):
    y_test_ = sparse2array_inv_binarize(y_test)
    num_predictions = [len(item) for item in y_test_]
    probabilities = classifier.predict_proba(X_test)
    sorted_indices_probs = probabilities.argsort()
    preds = [sorted_indices[-num:].tolist() for (sorted_indices, num) in zip(sorted_indices_probs, num_predictions)]
    return preds


################################################################################
def compute_metrics(y_test, preds):
    # range(y_test.shape[1])
    mlb = MultiLabelBinarizer()
    mlb.fit(preds)
    preds = mlb.transform(preds)
    # convert y_test from sparse back and forth to get binarized version.
    # There's probably a better way to do this
    y_test = sparse2array_inv_binarize(y_test)
    y_test = mlb.transform(y_test)

    microF1 = f1_score(y_test, preds, average='micro')
    macroF1 = f1_score(y_test, preds, average='macro')
    return microF1, macroF1


################################################################################
def plot_graph(train_size, res):
    micro, = plt.plot(train_size, [a[0] for a in res], c='b', marker='s', label='Micro F1')
    macro, = plt.plot(train_size, [a[1] for a in res], c='r', marker='x', label='Macro F1')
    plt.legend(handles=[micro, macro])
    plt.grid(True)
    plt.xlabel('Training Size')
    plt.ylabel('F1 score')
    plt.show()
    return


################################################################################
def evaluate(G, subs_coo, word_vec):
    classifiers = {'Logistic_Regression': OneVsRestClassifier(LogisticRegression())}

    features_matrix = np.asarray([word_vec[str(node)] for node in range(len(G.nodes()))])
    training_set_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_results = {}
    for (name, model) in classifiers.items():
        results = []
        for training_size in training_set_size:
            X_train, X_test, y_train, y_test = train_test_split(features_matrix, subs_coo, train_size=training_size,
                                                                random_state=42)
            model.fit(X_train, y_train)

            preds = custom_predict(model, X_test, y_test)
            microF1, macroF1 = compute_metrics(y_test, preds)

            results.append((microF1, macroF1))

        all_results[name] = results
    return all_results


def main():
    parser = argparse.ArgumentParser("DeepWalk", description="Implementation of " +
                                                             "DeepWalk model. File Author: Apoorva")
    parser.add_argument("--d", default=128, type=int, help="Dimensions of word embeddings")
    parser.add_argument("--walks", default=10, type=int, help="Number of walks per node")
    parser.add_argument("--len", default=30, type=int, help="Length of random walk")
    parser.add_argument("--window", default=5, type=int, help="Window size for skipgram")
    parser.add_argument("--hs", default=1, type=int, help="0 - Negative Sampling  1 - Hierarchical Softmax")
    parser.add_argument("--file", required=True, type=str, help="test_file")

    warnings.filterwarnings("ignore")

    cmdargs = "--d 100 --walks 15 --len 30 --window 5 --file file/blogcatalog.mat"
    args = parser.parse_args(cmdargs.split())
    process(args)


if __name__ == '__main__':
    main()
