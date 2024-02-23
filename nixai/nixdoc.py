import uuid

from typing import Any, List, Optional
from argparse import ArgumentParser

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import BaseStore
from langchain_core.pydantic_v1 import Field

from langchain.text_splitter import TextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from langchain_community.storage.sql import SQLDocStore

from ragatouille import RAGPretrainedModel

from nixai.settings import docstore_conn_str


collection_name = "nixdoc"


class ParentDocumentRetriever(BaseRetriever):
    rag: RAGPretrainedModel
    """Ragatouille rag to use"""

    index_name: str
    """Index name to use"""

    child_splitter: TextSplitter
    """The text splitter to use to create child documents."""

    id_key: str = "id"
    """The key to use to track the parent id. This will be stored in the
    metadata of child documents."""

    parent_splitter: Optional[Any] = None
    """The text splitter to use to create parent documents.
    If none, then the parent documents will be the raw documents passed in."""

    docstore: BaseStore[str, Document]
    """The storage interface for the parent documents"""

    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""

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

        docs: List[Document] = []
        full_docs = []
        for i, doc in enumerate(documents):
            _id = doc_ids[i]
            sub_docs = self.child_splitter.split_documents([doc])
            for _doc in sub_docs:
                _doc.metadata[self.id_key] = _id
            docs.extend(sub_docs)
            full_docs.append((_id, doc))

        # index sub documents to vectorstore
        self.rag.index(
            [d.page_content for d in docs],
            document_metadatas=[d.metadata for d in docs],
            index_name=self.index_name)

        # add documents to document store
        if add_to_docstore:
            self.docstore.mset(full_docs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        search_result = self.rag.search(query, **self.search_kwargs)

        sub_docs = [
            Document(
                page_content=doc["content"], metadata=doc.get("document_metadata", {})
            )
            for doc in search_result
        ]

        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])

        docs = self.docstore.mget(ids)

        return [d for d in docs if d is not None]

store = SQLDocStore(docstore_conn_str, collection_name=f"chroma/{collection_name}") 

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")],
    strip_headers=False)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=180, chunk_overlap=10)

retriever = ParentDocumentRetriever(
    docstore=store,
    rag=RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/"+collection_name),
    parent_splitter=md_splitter,
    child_splitter=text_splitter,
    index_name=collection_name,
    search_kwargs={"k": 3},
)

def embed_documents(path, **_):
    from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader

    # load all markdown files from directory
    loader = DirectoryLoader(path, glob="**/*.md", show_progress=True, loader_cls=UnstructuredMarkdownLoader, silent_errors=True)

    print(f"Loading documents")

    documents = loader.load()

    print(f"Embedding {len(documents)} documents")

    retriever.add_documents(documents)

def query_documents(query, **_):
    for doc in retriever.invoke(query):
        print("========================")
        print(doc.page_content)

if __name__ == "__main__":
    # Initialize the ArgParser with a description
    parser = ArgumentParser(description='nixdoc management script')

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
