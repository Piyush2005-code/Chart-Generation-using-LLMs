from openai import OpenAI
from dotenv import load_dotenv
import os
from fastapi import FastAPI, Query
import requests

app = FastAPI()
load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=QWEN_API_KEY,
)

# text = input("Enter the text you want to summarize: ")


# @app.get("/get_summary")
def get_summary(text: str):
    chat_history_for_summary_generation= [
    {"role": "system", "content": """You are an expert summarization model trained to process and condense large volumes of text while preserving the full meaning, logical flow, and contextual depth of the original material.

Your task is to generate a **faithful, coherent, and information-rich summary** of a long corpus of text. You must ensure that:
- No critical facts, nuances, or relationships between ideas are lost.
- The summary accurately reflects the author’s intent, tone, and argument structure.
- Contextual dependencies (e.g., pronouns, references, temporal order) remain consistent.
- Redundant, repetitive, or filler content is omitted.
- If the corpus is very long, first structure your summary hierarchically:
  1. Identify the **main themes or sections**.
  2. Summarize each section clearly.
  3. Then generate a **final integrated summary** that connects them cohesively.
- Do **not** hallucinate or infer beyond what is stated in the text.
- Maintain objectivity — do not insert personal opinions or biases.

When summarizing, your priority is to **compress the text without compromising meaning**.

If asked for a specific format (e.g., bullet points, paragraph form, or abstract style), adapt your summary accordingly.
"""},
    {"role": "user", "content": f"{text}\n\nSummarize the above text."},
]

    chat_completion_for_summary = client.chat.completions.create(
        model="qwen/qwen-2.5-7b-instruct",
        messages=chat_history_for_summary_generation,
        )
    response = chat_completion_for_summary.choices[0].message.content
    # print(response)
    return response

def get_mermaid_markdown(summary: str):

    chat_history_for_summary_generation= [
    {"role": "system", "content": """You are an expert summarization model trained to process and condense large volumes of text while preserving the full meaning, logical flow, and contextual depth of the original material.

Your task is to generate a **faithful, coherent, and information-rich summary** of a long corpus of text. You must ensure that:
- No critical facts, nuances, or relationships between ideas are lost.
- The summary accurately reflects the author’s intent, tone, and argument structure.
- Contextual dependencies (e.g., pronouns, references, temporal order) remain consistent.
- Redundant, repetitive, or filler content is omitted.
- If the corpus is very long, first structure your summary hierarchically:
  1. Identify the **main themes or sections**.
  2. Summarize each section clearly.
  3. Then generate a **final integrated summary** that connects them cohesively.
- Do **not** hallucinate or infer beyond what is stated in the text.
- Maintain objectivity — do not insert personal opinions or biases.

When summarizing, your priority is to **compress the text without compromising meaning**.

If asked for a specific format (e.g., bullet points, paragraph form, or abstract style), adapt your summary accordingly.
"""},
    {"role": "user", "content": f"{summary}\n\nSummarize the above text."},
]

    chat_completion_for_summary = client.chat.completions.create(
        model="qwen/qwen-2.5-7b-instruct",
        messages=chat_history_for_markdown_generation,
        )


# if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run("server:app", host="0.0.0.0", port=3525, reload=True)


