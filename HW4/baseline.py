#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DataScienceHW 
@File    ：baseline.py
@IDE     ：PyCharm 
@Author  ：Yaser
@Date    ：2021/10/29 20:10 
@Describe: 
"""
import random

import networkx as nx
import math
import matplotlib.pyplot as plt


def adamic_adar_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)

    def predict(u, v):
        return sum(1 / math.log(G.degree(w)) for w in nx.common_neighbors(G, u, v))

    return ((u, v, predict(u, v)) for u, v in ebunch)


# %%

def common_neighbors(G, u, v):
    if u not in G:
        raise nx.NetworkXError('u is not in the graph.')
    if v not in G:
        raise nx.NetworkXError('v is not in the graph.')

    sum = 0
    for w in G[u]:
        if w in G[v] and w not in (u, v):
            sum += 1
    return sum


# %%

def jaccard_coefficient(G, ebunch=None):
    def predict(u, v):
        union_size = len(set(G[u]) | set(G[v]))
        if union_size == 0:
            return 0
        return len(list(nx.common_neighbors(G, u, v))) / union_size

    if ebunch is None:
        ebunch = nx.non_edges(G)
    return ((u, v, predict(u, v)) for u, v in ebunch)


# %%

def resource_allocation_index(G, ebunch=None):
    def predict(u, v):
        return sum(1 / G.degree(w) for w in nx.common_neighbors(G, u, v))

    if ebunch is None:
        ebunch = nx.non_edges(G)
    return ((u, v, predict(u, v)) for u, v in ebunch)


if __name__ == '__main__':
    node_num = 20
    G = nx.Graph()  # 建立无向图
    H = nx.path_graph(node_num)  # 添加节点
    G.add_nodes_from(H)  # 添加节点
    i = 0


    def rand_edge(vi, vj, p=0.2):  # 默认概率p=0.1
        probability = random.random()  # 生成随机小数
        if probability < p:  # 如果小于p
            G.add_edge(vi, vj)  # 连接vi和vj节点


    while i < node_num:
        j = 0
        while j < i:
            rand_edge(i, j)  # 调用rand_edge()
            j += 1
        i += 1
    nx.draw_networkx(G, with_labels=True)
    plt.show()

    print(f"{20 * '='}adamic_adar_index{20 * '='}")
    for u, v, p in adamic_adar_index(G, G.edges()):
        if p > 0:
            print(u, v, "adamic_adar_index: ", p)

    print(f"{20 * '='}common_neighbors{20 * '='}")
    for i in range(node_num):
        for j in range(i + 1, node_num):
            nums = common_neighbors(G, i, j)
            if nums > 0:
                print(f"node {i} and {j} has common neighbors {nums}")

    print(f"{20 * '='}resource_allocation_index{20 * '='}")
    for u, v, p in resource_allocation_index(G, G.edges()):
        if p > 0:
            print(u, v,"resource_allocation_index: ",  p)
