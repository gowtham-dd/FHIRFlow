from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# import helper functions
from helper.helper import (
    load_pdf_files,
    filter_to_minimal_docs,
    download_embeddings,
    text_split
)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

print("Pinecone key loaded:", PINECONE_API_KEY is not None)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Load PDF documents
print("Loading PDF files...")
extracted_data = load_pdf_files("data")

# Clean metadata
minimal_docs = filter_to_minimal_docs(extracted_data)

# Split into chunks
print("Splitting documents...")
texts_chunk = text_split(minimal_docs)

# Load embedding model (384 dimension)
print("Loading embedding model...")
embeddings = download_embeddings()

# Pinecone index name
index_name = "fhirdb"

# Delete old index if dimension mismatch
if pc.has_index(index_name):
    print("Deleting old index...")
    pc.delete_index(index_name)

# Create new index with correct dimension
print("Creating new Pinecone index...")
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Connect to index
index = pc.Index(index_name)

# Upload vectors
print("Uploading document vectors to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=index_name
)

print("  Documents successfully stored in Pinecone!")