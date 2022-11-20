# Only need to run this once - script to download all the data and put it in a pandas DataFrame wga_df.csv
# TODO: precompute all the BERT embeddings and save them so they don't need to be recomputed when you train a model
# Run the create_wga_database.py script after running this to finish setup.

import os, sys, pandas as pd, numpy as np
from tqdm import tqdm


#### WGA Data Download #####

os.system("wget https://github.com/jrgillick/WGA-Data/raw/main/wga_catalog_with_descriptions.txt.zip")
os.system("unzip wga_catalog_with_descriptions.txt.zip")
wga_df = pd.read_csv('catalog_with_descriptions.txt', sep="\t", na_filter=False)
del(wga_df['Unnamed: 0'])


os.system("wget https://github.com/jrgillick/WGA-Data/raw/main/bio_catalog_with_text_bios.txt")
bio_df = pd.read_csv('bio_catalog_with_text_bios.txt', sep="\t", na_filter=False)
del(bio_df['Unnamed: 0'])


wga_df = pd.merge(wga_df, bio_df, how="left", left_on='AUTHOR', right_on='ARTIST')

os.system("wget https://github.com/jrgillick/WGA-Data/raw/main/bio_description_fuzzy_matches.txt")
bio_description_fuzzy_matches = open('bio_description_fuzzy_matches.txt').read().split('\n')
bio_description_fuzzy_matches = [int(t) for t in bio_description_fuzzy_matches]
wga_df['bio_description_fuzzy_match'] = bio_description_fuzzy_matches

os.system("rm -r __MACOSX")
os.system("rm bio_catalog_with_text_bios.txt")
os.system("rm bio_description_fuzzy_matches.txt")
os.system("rm catalog_with_descriptions.txt")
os.system("rm wga_catalog_with_descriptions.txt.zip")

# Save to wga_df.csv
wga_df.to_csv("wga_data.csv", sep='\t')

