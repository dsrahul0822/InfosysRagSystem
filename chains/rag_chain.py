from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def build_rag_chain(retriever):
    """
    Build a RAG chain using ChatOpenAI and provided retriever. 
    """
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_template("""
    You are an HR Assistant. Answer the question strictly using the context below. If the answer
    is not present, say "I don't know".
    context: {context}
    question: {question}
    """)

    chain = ({"context":retriever,
              "question":RunnablePassthrough()}
              | prompt | llm | StrOutputParser())
    
    return chain