from openai import OpenAI
from dotenv import load_dotenv
import os
from fastapi import FastAPI, Query
import requests
from google import genai
from google.genai import types
import subprocess


app = FastAPI()
load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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

# def get_mermaid_markdown(summary: str):

#     chat_history_for_summary_generation= [
#     {"role": "system", "content": """You are an expert summarization model trained to process and condense large volumes of text while preserving the full meaning, logical flow, and contextual depth of the original material.

# Your task is to generate a **faithful, coherent, and information-rich summary** of a long corpus of text. You must ensure that:
# - No critical facts, nuances, or relationships between ideas are lost.
# - The summary accurately reflects the author’s intent, tone, and argument structure.
# - Contextual dependencies (e.g., pronouns, references, temporal order) remain consistent.
# - Redundant, repetitive, or filler content is omitted.
# - If the corpus is very long, first structure your summary hierarchically:
#   1. Identify the **main themes or sections**.
#   2. Summarize each section clearly.
#   3. Then generate a **final integrated summary** that connects them cohesively.
# - Do **not** hallucinate or infer beyond what is stated in the text.
# - Maintain objectivity — do not insert personal opinions or biases.

# When summarizing, your priority is to **compress the text without compromising meaning**.

# If asked for a specific format (e.g., bullet points, paragraph form, or abstract style), adapt your summary accordingly.
# """},
#     {"role": "user", "content": f"{summary}\n\nSummarize the above text."},
# ]

#     chat_completion_for_summary = client.chat.completions.create(
#         model="qwen/qwen-2.5-7b-instruct",
#         messages=chat_history_for_markdown_generation,
#         )


# if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run("server:app", host="0.0.0.0", port=3525, reload=True)



def gemini_inference(summary: str,
                     temperature: float = 0.0,
                     max_output_tokens: int = 512,
                     thinking_budget: int = None) -> str:
    """
    Send a prompt to Gemini 2.5 Flash and return the generated text response.
    """
    client = genai.Client()
    
    contents = [
        f"""
You are an intelligent assistant that converts textual summaries or paragraphs into Mermaid markdown diagrams.

Your goal:
Analyze the input summary, identify the relationships, structure, or sequence it describes, 
and generate the most appropriate Mermaid markdown chart to visualize it.

---

### Instructions:

1. **Understand the input text**
   - Identify whether it describes a process, hierarchy, relationship, sequence, or data distribution.

2. **Choose the most appropriate Mermaid chart type**
   - Use:
     - `flowchart` → for processes, decisions, or logical flows
     - `sequenceDiagram` → for event or actor interactions over time
     - `classDiagram` → for structured entities, components, or systems
     - `mindmap` → for conceptual or topic hierarchies
     - `gantt` → for timelines or project stages
     - `pie` or `bar` → for numerical or category distributions

3. **Output format**
   - Begin with one line stating which chart type you chose and why.
   - Then output valid Mermaid markdown in a fenced code block:
     ```mermaid
     <chart_type>
       <chart_content>
     ```
   - Ensure Mermaid syntax correctness and clear node labels.
   - Do not include explanations inside the code block.

---

### Example Outputs

#### Example 1: (Process Summary)
**Input:** “The research workflow begins with collecting data, processing it, validating results, and then publishing the final paper.”

**Output:**
The text describes a process, so a flowchart best fits.

```mermaid
flowchart TD
    A[Collect Data] --> B[Process Data]
    B --> C[Validate Results]
    C --> D[Publish Paper]
```

```mermaid
mindmap
  root((Artificial Intelligence))
    Machine Learning
    Natural Language Processing
    Computer Vision
```
    
```mermaid
sequenceDiagram
    participant User
    participant Server
    participant Database
    User->>Server: Send Request
    Server->>Database: Query Data
    Database-->>Server: Return Results
    Server-->>User: Send Response
```
 
Now generate a Mermaid markdown for the following summary:
"f{summary}"
"""
    ]

    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens
    }
    if thinking_budget is not None:
        generation_config["thinking_budget"] = thinking_budget

    client = genai.Client(api_key="YOUR_API_KEY")

    response = client.models.generate_content(
    model="gemini-2.5-flash", contents=contents
    )
    print(response.text)
    
    candidate = response.candidates[0]
    return candidate.text


def save_chart(markdown: str):
    with open("temp_chart.mmd", "w") as f:
        f.write(markdown)

    cmd1 = ["mmdc", "-i", "temp_chart.mmd", "-o", "temp_chart_default.png", "temp_chart.svg", "-t", "default"]
    cmd2= ["mmdc", "-i", "temp_chart.mmd", "-o", "temp_chart_dark.png", "temp_chart.svg", "-t", "dark"]
    cmd3 = ["mmdc", "-i", "temp_chart.mmd", "-o", "temp_chart_forest.png", "temp_chart.svg", "-t", "forest"]
    cmd4 = ["mmdc", "-i", "temp_chart.mmd", "-o", "temp_chart_neutral.png", "temp_chart.svg", "-t", "neutral"]

    subprocess.run(cmd1, check=True)
    subprocess.run(cmd2, check=True)
    subprocess.run(cmd3, check=True)
    subprocess.run(cmd4, check=True)


if __name__ == "__main__":
    text = input("Enter the text of summary : ")

    summary = get_summary(text=text)

    mmd = gemini_inference(
        summary = summary,
        temperature=0.3,
        max_output_tokens=400,
        thinking_budget=1000
    )

    save_chart(mmd)

    print(mmd)
