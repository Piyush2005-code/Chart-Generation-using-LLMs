# Install required packages if needed
# pip install langchain[google-genai]

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from pdf_scanner_alternative import extract_with_pdfplumber

# If using environment variable for API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCD2LStsBJXjXNtysgp9Ma3hrCCmopRo-4"

# Initialize Gemini model (set to use Gemini, adjust model name as needed)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# The prompt template for diagram-relevant summary
prompt = ChatPromptTemplate.from_template(
    "From the given passage, produce a structured very concise summary, such that the summary contains only the most relevant nodes (people, objects, steps) and the connections (actions, dependencies, sequences) important for drawing a Mermaid diagram.\n\n{context}"
)



pdf_path = "Chart-Generation-using-LLMs/docs/doc1.pdf"  # Replace with your PDF file path
texts = extract_with_pdfplumber(pdf_path)

# Example input: Replace with your own text
input_text = '\n'.join(texts)  # Combine all pages' text into one string


# Convert the text to LangChain Document format
documents = [Document(page_content=input_text)]

# Create summarization chain
chain = create_stuff_documents_chain(llm, prompt)

# Run the summarization chain
result = chain.invoke({"context": documents})

print("Summary for diagram generation:")
print(result)
