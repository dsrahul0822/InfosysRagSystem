from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_document(documents):
    """
    Split document into chunks for embedding
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                              chunk_overlap=50)
    split_docs = splitter.split_documents(documents)
    return split_docs