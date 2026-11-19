"""Leaf photo -> disease (fine-tuned ViT, classify.py) -> 3 advice points (GPT-4o vision)."""

import base64
import os
import sys

from openai import OpenAI

from classify import classify

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
disease = classify(sys.argv[1])
img = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
r = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": [
    {"type": "text", "text": f"The leaf shows {disease}. Give a farmer 3 practical advice bullet points."},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}]}])
print(f"Detected: {disease}")
print(r.choices[0].message.content)
