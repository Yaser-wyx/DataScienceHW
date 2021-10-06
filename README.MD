# DataScience HomeWork

## HW1 Text Similarity

Solution: Word Embedding + Cosine Similarity Implementation Steps:

- Get each word's token id and attention_mask by using BertTokenizer
- Input the token id into the BertModel and then get each word's embedding
- Use `attention_mask*embedding` to get the masked embedding, it will mask the words which are shouldn't be attended to
- Sum all the words' embedding which is in one sentence, and calculate the mean of them, then get the embedding for one
  sentence
- Calculate the cosine similarity between each sentence

Result:
![](HW1_result.jpg)