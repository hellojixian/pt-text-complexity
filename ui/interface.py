import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.text_complixity import complexity_score
import gradio as gr
import json

def generate_html(input_text):
    # Generate HTML code dynamically based on the input
  score, features = complexity_score(input_text)
  html_code = f"<h1>PT Score: {score}!</h1>"
  # render the feature json as html
  html_code += f"<h2>Features:</h2><pre>{json.dumps(features, indent=4)}</pre>"
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