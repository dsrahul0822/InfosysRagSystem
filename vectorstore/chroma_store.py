from langchain_community.vectorstores import Chroma

def create_chroma_store(chunks, embeddings, persist_dir="chroma_db"):
    """
    Create and Persist Chroma vector store
    """
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb


def load_chroma_store(embeddings, persist_dir="chroma_db"):
    """
    Load existing chorma vector store
    """
    return Chroma(embedding_function=embeddings, persist_directory=persist_dir)