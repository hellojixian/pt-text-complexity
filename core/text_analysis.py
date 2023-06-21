import nltk
import re
import json
import string
import pandas as pd
from nltk.corpus import cmudict
import stanza

try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')

try:
  nltk.data.find('corpora/cmudict')
except:
  nltk.download('cmudict')

nlp = None
PARARE = re.compile('\n\n+')
CORPUS = pd.read_csv('data/corpus.csv', index_col='unique_word')

stemmer = nltk.stem.PorterStemmer()
cmu_dict = cmudict.dict()

# assume that the known word is a common word
default_word_weight = CORPUS['weights'].quantile(0.03)
# print(f"Default word weight: {default_word_weight}")

def complexity_measurement(text, with_grammar=True):
  if len(text) == 0:
    raise ValueError("I can't do this, there's no words there!")

  # check how many paragraphs
  words = nltk.word_tokenize(text.translate(str.maketrans("", "", string.punctuation)))
  sentences = nltk.sent_tokenize(text)
  grammars = analysis_grammar_structure(sentences) if with_grammar else []

  total_paragraphs = sum(1 for _ in PARARE.finditer(text)) + 1
  total_sentences = len(sentences)
  total_words = len(words)
  total_characters = sum(len(word) for word in words)
  total_weights = sum(CORPUS.loc[word, 'weights'] if word in CORPUS.index else default_word_weight for word in words)
  total_stems = sum(CORPUS.loc[word, 'stem'] if word in CORPUS.index else len(stemmer.stem(word)) for word in words)
  total_syllables = sum(CORPUS.loc[word, 'syllables'] if word in CORPUS.index else len(cmu_dict.get(word.lower(), [[]])[0]) for word in words)
  total_grammars = len(grammars)

  result = {
    "total":{
      "paragraphs": total_paragraphs,
      "sentences": total_sentences,
      "words": total_words,
      "characters": total_characters,
      "weights": total_weights,
      "stems": total_stems,
      "syllables": total_syllables,
      "grammar": total_grammars,
    },
    "average":{
      "sentences_per_paragraph": total_sentences / total_paragraphs,
      "words_per_sentence": total_words / total_sentences, # has feature
      "syllables_per_word": total_syllables / total_words, # has feature
      "stems_per_word": total_stems / total_words, # has feature
      "characters_per_word": total_characters / total_words, # has feature
    },
    "features": {
      "weights_per_word": (total_weights / total_words),
      "sentences_per_paragraph": (total_sentences / total_paragraphs) / 30,
      "words_per_sentence": (total_words / total_sentences) / 25,
      "syllables_per_character": (total_syllables / total_words) / 10,
      "stems_per_character": (total_stems / total_words) / 20,
      "characters_per_word": (total_characters / total_words) / 20,
      "grammars": (total_grammars) / 100,
    }
  }
  return result

def analysis_grammar_structure(sentences) -> list:
  # return []
  global nlp
  if nlp is None:
    # initialize the pipeline
    nlp = stanza.Pipeline(lang='en',
                      use_gpu=True,
                      verbose=False,
                      download_method=stanza.DownloadMethod.REUSE_RESOURCES,
                      processors='tokenize,pos,lemma,depparse')
  # set to store the unique dependency relations
  unique_relations = set()

  # iterate over the sentences
  for sentence in sentences:
      # process the sentence
      doc = nlp(sentence)
      # get the dependencies from the first sentence in the document
      dependencies = doc.sentences[0].dependencies

      # iterate over the dependencies and add the relation to the set
      for dep in dependencies:
          unique_relations.add(dep[1])

  return list(unique_relations)
