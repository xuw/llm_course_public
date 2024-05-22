import gradio as gr
import json
from openai import OpenAI

import os
vllmkey = os.environ["VLLM_API_KEY"]
modelname = os.environ["VLLM_MODEL_NAME"]
openai_url = os.environ["OPENAI_URL"]

client = OpenAI(
    base_url=openai_url,
    api_key=vllmkey,
)
model=modelname

def predict(message, history):

#### Your Task ####
# Insert code here to perform the inference

    history_openai_format = []
    history_openai_format.append(
        {"role": "system", "content": "You are chatting with an AI assistant. The assistant is helpful, creative, clever, and very friendly."})
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})
  
    print(history_openai_format)


    completion = client.chat.completions.create(
        model=model,
        messages=history_openai_format,
    )

    return completion.choices[0].message.content

#### End Task ####

gr.ChatInterface(predict).launch(server_name="0.0.0.0")
