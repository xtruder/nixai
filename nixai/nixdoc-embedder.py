from langchain.document_loaders.html import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from nixai.nixdoc import retriever


loader = UnstructuredHTMLLoader("doc.html")
documents = loader.load()

print("starting indexing")

retriever.add_documents(documents)

# main indexing method
#index(documents, record_manager, big_chunks_retriever, batch_size=10)
