import base64
import os
import sys

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
img = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
r = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": [
    {"type": "text", "text": "Leaf disease advice for a farmer, 3 bullet points."},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}]}])
print(r.choices[0].message.content)
