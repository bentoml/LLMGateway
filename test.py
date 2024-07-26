import os
from openai import OpenAI

BASE_URL = os.environ.get("BASE_URL", "http://localhost:3000/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "n/a")
PROMPT = os.environ.get("PROMPT", "Explain superconductors like I'm five years old in 500 words")
STREAM = True if os.environ.get("STREAM") == "1" else False
MODEL = os.environ.get("MODEL", "gpt-3.5-turbo")

client = OpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY)

chat_completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": PROMPT,
        }
    ],
    stream=STREAM,
)

if STREAM:
    for chunk in chat_completion:
        # Extract and print the content of the model's reply
        print(chunk.choices[0].delta.content or "", end="")
else:
    print(chat_completion.choices[0].message.content)
