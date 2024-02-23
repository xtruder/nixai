import argparse
import uuid

from typing import Any, List, Optional

from langchain_core.documents import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language, MarkdownHeaderTextSplitter
from langchain.retrievers import MultiVectorRetriever

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.storage.sql import SQLDocStore

from nixai.settings import chroma_dir, docstore_conn_str

collection_name = "nixdoc"

embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    max_length=768, threads=16,
    cache_dir="/tmp/fastembedding")

# load from disk
vectorstore = Chroma(
    persist_directory=chroma_dir,
    embedding_function=embeddings,
    collection_name=collection_name)

store = SQLDocStore(docstore_conn_str, collection_name=f"chroma/{collection_name}") 

class ParentDocumentRetriever(MultiVectorRetriever):
    child_splitter: Any
    """The text splitter to use to create child documents."""

    """The key to use to track the parent id. This will be stored in the
    metadata of child documents."""
    parent_splitter: Optional[MarkdownHeaderTextSplitter] = None
    """The text splitter to use to create parent documents.
    If none, then the parent documents will be the raw documents passed in."""

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        add_to_docstore: bool = True,
    ) -> None:
        """Adds documents to the docstore and vectorstores.

        Args:
            documents: List of documents to add
            ids: Optional list of ids for documents. If provided should be the same
                length as the list of documents. Can provided if parent documents
                are already in the document store and you don't want to re-add
                to the docstore. If not provided, random UUIDs will be used as
                ids.
            add_to_docstore: Boolean of whether to add documents to docstore.
                This can be false if and only if `ids` are provided. You may want
                to set this to False if the documents are already in the docstore
                and you don't want to re-add them.
        """
        if self.parent_splitter is not None:
            docs = []
            for doc in documents:
                splitted_docs = self.parent_splitter.split_text(doc.page_content)
                for d in splitted_docs:
                    d.metadata = doc.metadata
                docs.extend(splitted_docs)
            documents = docs

            print(f"spliited docs {len(documents)}")
        if ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in documents]
            if not add_to_docstore:
                raise ValueError(
                    "If ids are not passed in, `add_to_docstore` MUST be True"
                )
        else:
            if len(documents) != len(ids):
                raise ValueError(
                    "Got uneven list of documents and ids. "
                    "If `ids` is provided, should be same length as `documents`."
                )
            doc_ids = ids

        docs = []
        full_docs = []
        for i, doc in enumerate(documents):
            _id = doc_ids[i]
            sub_docs = self.child_splitter.split_documents([doc])
            for _doc in sub_docs:
                _doc.metadata[self.id_key] = _id
            docs.extend(sub_docs)
            full_docs.append((_id, doc))
        self.vectorstore.add_documents(docs)
        if add_to_docstore:
            self.docstore.mset(full_docs)


# separators = RecursiveCharacterTextSplitter.get_separators_for_language(Language.HTML)
parent_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")],
    strip_headers=False)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_type="similarity",
    search_kwargs={"k": 3},
)

# retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def embed_documents(path, **_):
    from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
    #from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    # load all markdown files from directory
    loader = DirectoryLoader(path, glob="**/*.md", show_progress=True, loader_cls=UnstructuredMarkdownLoader, silent_errors=True)
    # md_splitter = MarkdownHeaderTextSplitter(
    #     headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")])
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    print(f"Loading documents")

    documents = loader.load()

    # print(f"Splitting {len(documents)} documents")

    # all_docs = []
    # for doc in documents:
    #     docs = md_splitter.split_text(doc.page_content)
    #     docs = text_splitter.split_documents(docs)

    #     all_docs.extend(docs)

    print(f"Embedding {len(documents)} documents")

    #vectorstore.add_documents(all_docs)
    retriever.add_documents(documents)

def query_documents(query, **_):
    for doc in retriever.invoke(query):
        print("========================")
        print(doc.page_content)

if __name__ == "__main__":
    # Initialize the ArgParser with a description
    parser = argparse.ArgumentParser(description='nixdoc management script')

    # Create subparsers for the two different commands
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Define 'embed' command
    embed_parser = subparsers.add_parser('embed', help='Generate embeddings for documents')
    embed_parser.add_argument('path', type=str, help='File path to the document to embed')
    embed_parser.set_defaults(func=embed_documents)

    # Define 'query' command
    query_parser = subparsers.add_parser('query', help='Query documents')
    query_parser.add_argument('query', type=str, help='The query string to retrieve documents for')
    query_parser.set_defaults(func=query_documents)

    # Parse arguments from sys.argv
    args = parser.parse_args()

    # Check if a subcommand was provided
    if 'func' in args:
        args.func(**vars(args))
    else:
        parser.print_help()
