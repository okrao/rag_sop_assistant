import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq

# Load the .env file from the parent directory (ai-portfolio)
load_dotenv("../.env")

# Initialize Groq client with the defensive safeguard
raw_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=raw_key.strip() if raw_key else None)

print("1. Reading the SOP document...")
loader = TextLoader("hsa_policy.txt")
docs = loader.load()

print("2. Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

print("3. Converting text to vector embeddings and storing in ChromaDB...")
# This uses a fast, lightweight, open-source embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(chunks, embeddings)

print("4. Searching the database for the representative's question...")
question = "A customer wants to know if they can use their HSA for a gym membership to avoid the 20% penalty. What is the exact policy?"

# Retrieve the most relevant chunks of text from the database
results = vector_store.similarity_search(question, k=2)
context = "\n\n".join([doc.page_content for doc in results])

print("5. Generating AI response based ONLY on the retrieved SOP...\n")
system_prompt = f"""You are a strict compliance assistant for a contact center. 
Answer the user's question using ONLY the following standard operating procedure text. 
Do not use any outside knowledge.

SOP CONTEXT:
{context}"""

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ],
    model="openai/gpt-oss-120b",
    temperature=0.1
)

print("--- AI RAG RESPONSE ---")
print(chat_completion.choices[0].message.content)
print("-----------------------\n")