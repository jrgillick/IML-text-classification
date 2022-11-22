#### Setup BERT Stuff


#### SKLearn, nearest neighbors stuff Classifier stuff ####
import pandas as pd, numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier

import torch, transformers, torch.nn as nn
from transformers import BertTokenizer, BertTokenizerFast, BertModel

import time
from tqdm import tqdm
import logging
logger = logging.getLogger()
logger.disabled = True
#logging.getLogger("transformers").setLevel(logging.ERROR)

#Sets random seeds for reproducibility
seed=0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

#Check GPU/CPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on {}".format(device))

#Load tokenizer and pretrained BERT model from Huggingface
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', do_lower_case=False, do_basic_tokenize=False)
bert = BertModel.from_pretrained("bert-base-cased")
_ = bert.eval()

## Define some functions for getting text representations from BERT's encoder

# Get a BERT embedding for an entire chunk of text like a phrase/sentence/short paragraph
def get_bert_text_embedding(text):
  toks = tokenizer(text, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping=True)#, max_length=max_toks)
  batch_x = toks
  bert_output = bert(input_ids=batch_x["input_ids"],
                            attention_mask=batch_x["attention_mask"],
                            token_type_ids=batch_x["token_type_ids"],
                            output_hidden_states=False)

  bert_hidden_state = bert_output['last_hidden_state']

  # We're going to use the 'CLS' token at the *last* layer output (layer -1)
  out = bert_hidden_state[:,0,:]
  return out

def get_batch_bert_embeddings(text_list, batch_size):
  batches = []
  i = 0
  while i < len(text_list):
    batches.append(get_bert_text_embedding(text_list[i:i+batch_size]))
    i += batch_size
  return torch.cat(batches, dim=0)






#### Modeling Stuff


def makeSentenceDF(sentence_list, category, bert_sentence_embeddings):
    vector_list = [bert_sentence_embeddings[key] for key in sentence_list]
 
    # Zip the sentences together with their vector representations
    word_vec_zip = zip(sentence_list, vector_list)

    # Cast to a dict so we can turn it into a DataFrame
    word_vec_dict = dict(word_vec_zip)
    df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
    
    df2 = df.assign(label=category)
    return df2

def makeSentenceTestDF(testsentences, bert_sentence_embeddings):
    vector_list = [bert_sentence_embeddings[key] for key in testsentences]

    # Zip the words together with their vector representations
    word_vec_zip = zip(testsentences, vector_list)

    # Cast to a dict so we can turn it into a DataFrame
    word_vec_dict = dict(word_vec_zip)
    df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
    return df

def makeSentenceTrainingSet(sentences1, sentences2, category1, category2, bert_sentence_embeddings):
    df1 = makeSentenceDF(sentences1, category1, bert_sentence_embeddings)
    df2 = makeSentenceDF(sentences2, category2, bert_sentence_embeddings)
    frames = [df1, df2]
    df = pd.concat(frames)
    return df

def trainSentenceClassifier(sentences1, sentences2, category1, category2, bert_sentence_embeddings, k=3):
    # get embeddings if any are missing
    t0 = time.time()
    for sentence in sentences1 + sentences2:
      if sentence not in bert_sentence_embeddings:
        embedding = get_bert_text_embedding(sentence).squeeze().detach().numpy()
        bert_sentence_embeddings[sentence] = embedding
    print(f"Computed missing embeddings in {time.time()-t0} seconds")

    t0 = time.time()
    training = makeSentenceTrainingSet(sentences1, sentences2, category1, category2, bert_sentence_embeddings)
    knn = KNeighborsClassifier(n_neighbors=k)
    X = training.drop(['label'], axis=1)
    y = training['label']
    knn.fit(X, y)
    print(f"Trained knn in {time.time()-t0} seconds")
    return knn

def testSentenceClassifier(testsentences, knn, bert_sentence_embeddings):
    # get embeddings if any are missing
    t0 = time.time()
    for sentence in testsentences:
      if sentence not in bert_sentence_embeddings:
        embedding = get_bert_text_embedding(sentence).squeeze().detach().numpy()
        bert_sentence_embeddings[sentence] = embedding
    print(f"Computed missing embeddings in {time.time()-t0} seconds")

    testdf = makeSentenceTestDF(testsentences, bert_sentence_embeddings)
    preds = knn.predict(testdf)
    
    data = {'Sentence':testdf.index,
        'Predicted':preds}
  
    # Create DataFrame
    t = pd.DataFrame(data)
    return t


#### Spacy for sentence splitting

from spacy.lang.en import English

nlp = English()
nlp.add_pipe("sentencizer")
doc = nlp("This is a sentence. This is another sentence.")
assert len(list(doc.sents)) == 2

def get_sentences(text):
  doc = nlp(text)
  sents = list(doc.sents)
  return [str(s) for s in sents]


logger.disabled = False
