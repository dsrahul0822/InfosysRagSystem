def get_retriever(vectordb):
    """
    Convert vector store to retriever
    """

    return vectordb.as_retriever(search_kwargs={"k":3})
