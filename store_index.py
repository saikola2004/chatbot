from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY



file_path =r'C:\Users\satis\OneDrive\Desktop\Edunet_class\techskham\chatbot\chatbot\Data\medicalbook.pdf'


extracted_data = load_pdf(file_path)




text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()



pc = Pinecone(api_key=PINECONE_API_KEY)  # Use your real key
index_name = "medicalbot"
# Create index
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)


from langchain_pinecone import PineconeVectorStore
docsearch = PineconeVectorStore.from_documents(
    documents= text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

