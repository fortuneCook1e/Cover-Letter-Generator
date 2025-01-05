from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
import re
from langchain.schema import Document
import os


class Profile:
    def __init__(self, profile_path="./resource/profile.txt", vector_store_path="./data/vector_store"):
        # Initialize the embedding model once for reuse
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store_path = vector_store_path
        
        # Load or create vector store
        self.vector_store = self._initialize_vector_store(profile_path)
        
    def _initialize_vector_store(self, profile_path):
        persist_directory = self.vector_store_path
        
        # Check if the Chroma store directory exists and has files (indicating an existing store)
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            # Try to load the existing vector store
            try:
                vector_store = Chroma(persist_directory=persist_directory, embedding_function=self.embedding_model)
                print("Vector store loaded successfully.")
                return vector_store
            except Exception as e:
                print(f"Failed to load existing vector store: {e}")
        
        # If the directory is empty or doesn't exist, create a new vector store
        print("No existing vector store found. Creating a new one...")
        
        # Load the document and split into sections
        loader = TextLoader(profile_path)
        document = loader.load()
        document_text = document.pop().page_content
        
        # Split document text into structured sections
        sections = re.split(r"### (.+)", document_text)
        structured_sections = []
        for i in range(1, len(sections), 2):
            title = sections[i].strip()
            content = sections[i + 1].strip()
            structured_sections.append({"title": title, "content": content})
        
        # Convert each section into Document objects with metadata
        documents = [Document(page_content=section["content"], metadata={"title": section["title"]}) for section in structured_sections]
        
        # Create and return the new vector store
        vector_store = Chroma.from_documents(documents, self.embedding_model, persist_directory=persist_directory)
        return vector_store

    def retrieve_from_vectorstore(self, query, k=6):
        # Perform similarity search to retrieve relevant sections
        relevant_docs = self.vector_store.similarity_search(query, k=k)
        
        # Combine retrieved sections into a single string for LLM prompt
        profile_info = '\n'.join([f"{doc.metadata['title']}\n{doc.page_content}\n" for doc in relevant_docs])
        return profile_info
