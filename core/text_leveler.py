from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import nltk

try:
  nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
  nltk.download('averaged_perceptron_tagger')

try:
  nltk.data.find('corpora/wordnet.zip')
  nltk.data.find('corpora/omw-1.4.zip')
except LookupError:
  nltk.download('wordnet')
  nltk.download('omw-1.4')

import pandas as pd
import os


data_df = None
wnl = None
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = f"{project_root}/data/en_cefr_level.csv"

rank_mapping = {'A1':  [1, 600],
                'A2':  [601, 1200],
                'B1':  [1201, 2500],
                'B2':  [2501, 3000],
                'C1':  [3001, 10000],
                'C2':  [10001, 20000],
                'C2P': [20001, 50000]}

def extract_tokens(text:str) -> list:
  global data_df, wnl
  if data_df is None: data_df = pd.read_csv(data_path)
  if wnl is None: wnl = WordNetLemmatizer()
  words = word_tokenize(text)
  tokens = pos_tag(words)
  words = [[token[0],token[1],
            wnl.lemmatize(word=token[0],
                  pos=token[1][0].lower() if token[1][0].lower() in ['a', 'n', 'v', 'r' ,'s'] else 'n'),
   ] for token in tokens]
  return words

def word_leveling(word:str) -> dict:
  global data_df, rank_mapping
  rec = data_df[data_df['Word'] == word]
  rank = rec['Rank'].values[0] if len(rec) > 0 else 0
  if rank == 0:
    level = 'UNK'
    score = 0
  else:
    level = [level for level, rank_range in rank_mapping.items() if rank_range[0] <= rank <= rank_range[1]][0]
    score = (rank - rank_mapping[level][0]) / (rank_mapping[level][1] - rank_mapping[level][0])
    score = round(score, 4)
  return {'level': level, 'score': score, 'rank': rank}

def text_leveling(text:str) -> dict:
  global data_df
  tokens = extract_tokens(text)
  output = { 'meta': {},'words': []}

  ranks = []
  rank_levels = dict([(level, 0) for level in rank_mapping.keys()])
  for token in tokens:
    [org_word, pos, lemmatized_word] = token
    word_level = word_leveling(lemmatized_word)
    output['words'].append({
      'word': str(org_word),
      'pos': str(pos),
      'rank': int(word_level['rank']),
      'level': str(word_level['level']),
      'score': float(word_level['score']),
    })
    if word_level['rank'] > 0: ranks.append(word_level['rank'])
    if word_level['level'] != 'UNK': rank_levels[word_level['level']] += 1
  output['meta']['rank_mean'] = round(sum(ranks) / len(ranks), 2) if len(ranks) > 0 else 0
  output['meta']['rank_q75'] = round(pd.Series(ranks).quantile(0.75), 2) if len(ranks) > 0 else 0
  output['meta']['rank_q50'] = round(pd.Series(ranks).quantile(0.5), 2) if len(ranks) > 0 else 0
  return output
