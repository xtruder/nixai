from os import getenv

chroma_dir = getenv("CHROMA_DIR") or "chroma_db"
docstore_conn_str = getenv("DOCSTORE_CONN_STRING") or "sqlite:///docstore.sql"
