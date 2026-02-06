from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_embeddings():
    """
    Initialize and return OpenaI embeddings
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return embeddings
