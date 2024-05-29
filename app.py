import gradio as gr
from sample import sample


title = 'Generating Names with a Character-Level RNN'

article = '''
Basic character-level Recurrent Neural Network (RNN) to generate names from languages.\n
The model was trained on a few thousand surnames from 18 languages of origin, and generates a name according to the specified language.
'''

gr.Interface(
    fn=sample,
    inputs=gr.Textbox(lines=1, label="Input language"),
    outputs="text",
    title=title,
    article=article,
    examples=[['Arabic'], ['Chinese'], ['Czech'], ['Dutch'], ['English'], ['French'], 
               ['German'], ['Greek'], ['Irish'], ['Italian'], ['Japanese'], ['Korean'], 
               ['Polish'], ['Portuguese'], ['Russian'], ['Scottish'], ['Spanish'], ['Vietnamese']],
    allow_flagging="never"
).launch()