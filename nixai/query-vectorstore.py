from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from nixai.nixdoc import retriever as nixdoc_retriever

collection_name = "nixdoc"

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", max_length=512, threads=16, cache_dir="/tmp/fastembedding")

# load from disk
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name=collection_name)

print(vectorstore.similarity_search("How to uinstall nix pacakge manager?", k=4))
