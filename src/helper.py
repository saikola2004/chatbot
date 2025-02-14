from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

#def load_multiple_pdfs(file_paths):
    #all_documents = []
    #for file_path in file_paths:
        #if not os.path.exists(file_path):  # Check if file exists
            #print(f"Error: File not found -> {file_path}")
            #continue  # Skip missing files

        #try:
            #loader = PyPDFLoader(file_path)
            #documents = loader.load()
            #all_documents.extend(documents)  # Append all pages from the PDF
        #except Exception as e:
           # print(f"Error loading {file_path}: {e}")  # Print any errors encountered

    #return all_documents
# Extract data from a single PDF file
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)  # Load a single file
    documents = loader.load()
    return documents

# Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
