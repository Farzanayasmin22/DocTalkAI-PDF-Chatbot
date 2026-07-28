from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    """
    Load a PDF document and return its pages as LangChain Document objects.
    """
    loader = PyPDFLoader(file_path)
    return loader.load()
