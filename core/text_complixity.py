import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.text_analysis as ta
import numpy as np
import pandas as pd

def feature_extract(text) -> dict:
  if len(text) == 0:
    raise ValueError("I can't do this, there's no words there!")
  return ta.complexity_measurement(text)

def normalize(features) -> np.array:
  embedding = pd.Series(features['features'], dtype='float')
  embedding = embedding.clip(lower=0)
  # print(json.dumps(features, indent=2))
  # print(embedding)
  return embedding

def scale_complexity_score(scores) -> float:
  scores = np.clip(np.tanh((scores - 0.35) * 1.6) * 1200, 0, 1200)
  return scores

def complexity_score(text) -> tuple[float, dict]:
  features = feature_extract(text)
  complexity = normalize(features)
  score = np.linalg.norm(complexity - np.zeros(complexity.shape[0]))
  complexityScore = scale_complexity_score(score)
  features['score'] = score
  return complexityScore, features