from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
API_KEY = os.environ.get('API_KEY')

# Load Hugging Face embeddings
embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

# Initialize Pinecone vector store
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Configure Gemini
genai.configure(api_key= API_KEY)
model = genai.GenerativeModel(model_name="gemini-pro")

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def rag_chain(input_text):
    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(input_text)
    retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant context available."

    # Get chat history
    history = memory.load_memory_variables({})
    chat_history = history.get("chat_history", [])

    # Format chat history
    chat_history_str = ""
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        else:
            role = msg.type
        chat_history_str += f"{role}: {msg.content}\n"

    # Create formatted input
    formatted_input = f"{system_prompt}\n\nChat History:\n{chat_history_str}\nContext:\n{retrieved_context}\n\nUser Query: {input_text}"

    # Generate response
    response = model.generate_content(formatted_input)
    response_text = response.text

    # Clean up the response
    # Remove extra dashes and ensure proper new lines
    cleaned_response = response_text.replace(" - - ", "\n").replace(" - ", "\n").strip()

    # Highlight important points and format as bullet points
    # Split the cleaned response into lines
    response_lines = cleaned_response.split("\n")

    # Format important points as bullet points and highlight them
    formatted_response = []
    for line in response_lines:
        if ":" in line or "include" in line or "symptoms" in line.lower():  # Identify important lines
            formatted_response.append(f"**{line.strip()}**")  # Highlight important lines
        else:
            formatted_response.append(f"- {line.strip()}")  # Format other lines as bullet points

    # Join the formatted lines into a single response
    formatted_response = "\n".join(formatted_response)

    # Update memory
    memory.save_context({"input": input_text}, {"output": formatted_response})

    return formatted_response

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    return rag_chain(msg)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)