import pandas as pd
from tqdm import tqdm
import pickle

##### pre-processing just for purposes of this demo: # only keep the images with 1-sentence text descriptions
wga_df = pd.read_csv("wga_data.csv", sep="\t", na_filter=False)
del(wga_df['Unnamed: 0'])

wga_df = wga_df.drop_duplicates('text_description', keep='first')
wga_df.reset_index(inplace=True, drop=True)

one_sentence_description = []
for s in list(wga_df.text_description_sentences): 
  if(len(s.split("**sb**")) == 1 and s!=''):
    one_sentence_description.append(True)
  else:
    one_sentence_description.append(False)

wga_df['one_sentence_description'] = one_sentence_description
wga_df = wga_df.drop_duplicates('text_description', keep='first')
wga_df = wga_df[wga_df['one_sentence_description']==True]
wga_df.reset_index(inplace=True, drop=True)

#shuffle
wga_df = wga_df.sample(frac=1, random_state=0)
wga_df.reset_index(inplace=True, drop=True)
#####

#Create database for Flask app
from app import db, Artwork, Classifier#, Label
import modeling 


def compute_and_save_bert_embeddings(sentences, embeddings_file):
  bert_sentence_embeddings = {}
  import time
  t0 = time.time()
  for sentence in tqdm(sentences):
    embedding = modeling.get_bert_text_embedding(s).squeeze().detach().numpy()
    bert_sentence_embeddings[sentence] = embedding
  print(time.time()-t0)
  with open(embeddings_file, 'wb') as f:
    pickle.dump(bert_sentence_embeddings, f)

def create_artwork_database():
  df = wga_df[wga_df['TYPE']=='landscape']
  df.reset_index(inplace=True, drop=True) 
  df = df[0:500]
  ids = list(df.index)
  descriptions = list(df.text_description_sentences)
  compute_and_save_bert_embeddings(descriptions, embeddings_file='bert_sentence_embeddings.pkl')
  for i in range(len(df)):
    artwork = Artwork(unique_id = ids[i],
                      id = ids[i],
                      description = descriptions[i],
                      labelA = None,
                      labelB = None,
                      predicted = None,
                      hide = None,
                      classifier_code='0')
    db.session.add(artwork)
    db.session.commit()
  print(f'Added {len(df)} artworks to the database.') 
  classifier = Classifier(id=0, classifier_code='qwertyqwe', categoryA='AAA', categoryB='BBB')
  db.session.add(classifier)
    

if __name__ == '__main__':
  create_artwork_database()
