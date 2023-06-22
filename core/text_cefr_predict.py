import os,sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
from cefr_predictor.inference import Model

model = None
model_file = f"{project_root}/lib/cefr_predictor/models/xgboost.joblib"

def cefr_predict(text:str):
  global model, model_file
  if model is None: model = Model(model_file)
  texts = [text]
  levels, scores = model.predict_decode(texts)
  for i, (text, level, score) in enumerate(zip(texts, levels, scores)):
    score = {k: round(v, 4) for k, v in score.items()}
    return level, score