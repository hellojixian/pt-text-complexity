import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.text_complixity import complexity_score
from core.text_leveler import text_leveling
from core.text_cefr_predict import cefr_predict

import gradio as gr
import json

def generate_css() -> str:
  css = """
  .cefr {margin: 1em 0; line-height: 24px;}
  .cefr .legend {margin: 0 0 0.5em 0; font-size: 0.8em;}
  .cefr span {padding: 0 1px; margin:0 1px; display: inline-block;}
  .cefr .a1 {background-color: #ff840090;}
  .cefr .a2 {background-color: #ffc00090;}
  .cefr .b1 {background-color: #8aff0090;}
  .cefr .b2 {background-color: #00ffd890;}
  .cefr .c1 {background-color: #0084ff90;}
  .cefr .c2 {background-color: #0000ff90;}
  .cefr .c2p {background-color: #c600ff90;}
  .cefr .unk {background-color: #ff000000;}

  """
  return css

def generate_cefr_colortext(words) -> str:
  html = """
  <p class='legend'>
  <strong>CEFR Level</strong>
  <span class='a1'>CEFR A1</span>
  <span class='a2'>CEFR A2</span>
  <span class='b1'>CEFR B1</span>
  <span class='b2'>CEFR B2</span>
  <span class='c1'>CEFR C1</span>
  <span class='c2'>CEFR C2</span>
  <span class='c2p'>CEFR C2+</span>
  </p>
  """
  for word in words:
    html += f"<span title='{word['score']}' class='{word['level'].lower()}'>{word['word']}</span>"
  return html

def generate_html(input_text):
    # Generate HTML code dynamically based on the input
  pt_score, features = complexity_score(input_text)
  cefr_level, cefr_score = cefr_predict(input_text)
  words = text_leveling(input_text)['words']
  html_code = f"<h1>PT Score: {pt_score:.0f}</h1>"
  html_code = f"<h2>CEFR Level: {cefr_level}</h2>"
  # render the feature json as html
  output = {
    'cefr_level': cefr_level,
    'cefr_text_score': cefr_score,
    'pt_score': pt_score,
    'pt_features': features,
    'words': words,
  }

  html_code += f"<style>{generate_css()}</style>"
  html_code += f"<h2>Features:</h2>"
  html_code += f"<div class='cefr'>{generate_cefr_colortext(words)}</div>"
  html_code += f"<pre>{json.dumps(output, indent=2)}</pre>"
  return html_code

def webapp():
  interface = gr.Interface(
      title="Pickatale Text Complexity Measurement",
      fn=generate_html,
      inputs=[gr.components.Textbox(lines=20,
                                    placeholder="Please enter your text here, and click Submit button to see the result.",
                                    label="Text",
                                    )],
      outputs="html",
      allow_flagging='never',

  )
  return interface