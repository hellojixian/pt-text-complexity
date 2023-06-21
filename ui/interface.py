import gradio as gr

def generate_html(input_text):
    # Generate HTML code dynamically based on the input
    html_code = f"<h1>Hello, {input_text}!</h1>"
    return html_code

def webapp():
  interface = gr.Interface(
      title="Pickatale Text Complexity Measurement",
      fn=generate_html,
      inputs=[gr.components.Textbox(lines=2,
                                    placeholder="Please enter your text here, and click Submit button to see the result.",
                                    label="Text",
                                    )],
      outputs="html",
      allow_flagging='never',

  )
  return interface