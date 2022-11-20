import pandas as pd

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
