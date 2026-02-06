from langchain_community.document_loaders import PyPDFLoader

def load_pdf(path: str):
    """
    Load a PDF document from the specified path 
    """
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents