import time

from tqdm import tqdm
from pathlib import Path
from typing import Iterable, Iterator, List, TypeVar
from together import RateLimitError

from langchain.docstore.document import Document
from langchain.document_loaders.text import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from langchain.indexes import index, SQLRecordManager

from langchain_together.embeddings import TogetherEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


T = TypeVar('T')

def batched(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    """
    Batch an iterable into lists of a specified size.

    :param iterable: An input iterable.
    :param batch_size: The size of each batch as an integer.
    :return: An iterator of batches, where each batch is a list of elements.
    """
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1")

    iterator = iter(iterable)
    while True:
        batch_items = list()
        try:
            for _ in range(batch_size):
                batch_items.append(next(iterator))
        except StopIteration:
            # End of the iterable reached
            if batch_items:
                yield batch_items
            break
        yield batch_items


class RateLimitTogetherEmbeddings(FastEmbedEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        while True:
            try:
                result = super().embed_documents(texts)
                #print("embedding success", texts)
                return result
            except RateLimitError:
                print("retrying embedding docs", len(texts))
                time.sleep(5)


def load_and_split_docs(nixpkgs_path = "/home/offlinehq/Code/nixpkgs") -> Iterator[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=2,
        separators=[
            "\nlet ",
            "\nif ",
            # Now split by the normal type of lines
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )

    print("starting loader:", nixpkgs_path)

    all_files = [p for p in Path(nixpkgs_path).rglob('**/*.nix')]

    for path in (pbar := tqdm(all_files)):
        try:
            with open(str(path.absolute()), encoding='utf-8') as f:
                text = f.read()
                chunks = text_splitter.split_text(text)

                relative_path = path.relative_to(nixpkgs_path)
                pbar.set_postfix_str(f"{relative_path}, chunks: {len(chunks)}")

                if len(chunks) > 5000:
                    continue

                for chunk in chunks:
                    yield Document(page_content=chunk, metadata={"source": str(relative_path)})

        except:
            print(f"error loading: {path.absolute()}")

collection_name = "nixpkgs"

#embeddings = RateLimitTogetherEmbeddings(threads=16)
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", max_length=512, threads=16)

# load from disk
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name=collection_name)

namespace = f"chromadb/{collection_name}"

record_manager = SQLRecordManager(namespace, db_url="sqlite:///record_manager_cache.sql")
record_manager.create_schema()

# main indexing method
index(load_and_split_docs(), record_manager, vectorstore, batch_size=10)
