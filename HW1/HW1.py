#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DataScienceHW 
@File    ：HW1.py
@IDE     ：PyCharm 
@Author  ：Yaser
@Date    ：2021/10/6 10:26 
@Describe: Using embedding + cosine to calculate sentence similarity.
"""
from loguru import logger
from matplotlib import pyplot as plt
from transformers import DistilBertModel, DistilBertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pandas.plotting import table
import matplotlib


def get_embeddings(model, tokenizer, sentences):
    # token_result contains input_ids and attention_mask
    # input_ids is the id in the vocabulary directory,
    # attention_mask is used to mask which is important
    # it(attention_mask) indicates to the model which tokens should be attended to, and which should not.
    inputs = tokenizer(sentences, return_tensors="pt", max_length=128, padding="max_length")
    outputs = model(**inputs)
    words_embedding = outputs.last_hidden_state
    masks = inputs["attention_mask"].unsqueeze(-1).expand(words_embedding.size()).float()
    # shape of masks: n*128*1024(n is the number of sentence)
    masked_words_embedding = words_embedding * masks  # mask the words which are shouldn't be attended to
    # shape of masked_words_embedding: n*128*1024
    # calculate the sentence's embedding by sum all of the word's embedding and take the mean of them
    sentences_embedding = torch.sum(masked_words_embedding, 1) / masks.sum(1)
    return sentences_embedding


def show_img_result(result_pd):
    # matplotlib.rcParams['font.size'] = 35  # set fontsize
    matplotlib.rc('font', size=200)
    matplotlib.rcParams['font.family'] = 'SimHei'  # set chinese font

    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111, frame_on=False)  # no visible frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis

    table(ax, result_pd, loc='center')  # where df is your data frame

    plt.savefig('HW1_result.jpg')  # save as image


def load_test_sentences(txt_path):
    test_sentences = []
    with open(txt_path, mode="r", encoding="utf-8") as file:
        for line in file.readlines():
            test_sentences.append(line.strip())  # remove the blank and \n
    return test_sentences


def main():
    # loading tokenizer and model
    logger.info("loading tokenizer and model....")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")

    model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
    # test sentences
    txt_path = "/data/wyx/project/DataScienceHW/HW1/TestSentences.txt"
    sentences = load_test_sentences(txt_path)
    # get sentences' embedding and convert them to numpy array
    logger.info("get embedding....")
    sentences_embedding = get_embeddings(model, tokenizer, sentences).detach().numpy()
    data = []
    # calculate the cosine_similarity for each sentence
    logger.info("calculate similarity....")
    for idx, sentence_embedding in enumerate(sentences_embedding):
        for idx2 in range(len(sentences_embedding)):
            cosine_similarity_result = cosine_similarity([sentence_embedding], [sentences_embedding[idx2]])[0][0]
            data.append([sentences[idx], sentences[idx2], f"{round(cosine_similarity_result * 100, 3)}%"])

    result_pd = pd.DataFrame(data=data, columns=("sentenceA", "sentenceB", "similarity"))
    logger.info("save as excel....")
    result_pd.to_excel("./HW1_result.xlsx")
    logger.info("show result....")
    show_img_result(result_pd)
    logger.info("finished!")


if __name__ == '__main__':
    main()
