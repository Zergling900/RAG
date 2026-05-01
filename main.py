import json
from pathlib import Path
from hashlib import md5

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.readers.file import PyMuPDFReader

from llama_index.vector_stores.milvus import MilvusVectorStore


from src.config import MILVUS_URI, MILVUS_COLLECTION, EMBED_DIM, DATA_DIR


TRACK_FILE = Path("ingested_files.json")


def file_hash(path: Path):
    return md5(path.read_bytes()).hexdigest()


def load_track():
    if TRACK_FILE.exists():
        return json.loads(TRACK_FILE.read_text())
    return {}


def save_track(data):
    TRACK_FILE.write_text(json.dumps(data, indent=2))


def get_track_key(path: Path):
    return path.relative_to(DATA_DIR).as_posix()


def iter_document_files():
    return sorted(file for file in Path(DATA_DIR).rglob("*") if file.is_file())


def get_new_documents():
    tracked = load_track()
    new_files = []
    updated_track = {}
    seen_hashes = set()

    for file in iter_document_files():
        h = file_hash(file)
        key = get_track_key(file)
        updated_track[key] = h

        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        if key not in tracked or tracked[key] != h:
            new_files.append(file)

    return new_files, updated_track


def build_or_update_index():
    vector_store = MilvusVectorStore(
        uri=MILVUS_URI,
        dim=EMBED_DIM,
        collection_name=MILVUS_COLLECTION,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    new_files, updated_track = get_new_documents()

    if not new_files:
        print("No new documents.")
        return VectorStoreIndex.from_vector_store(vector_store)

    print(f"Ingesting {len(new_files)} new files...")

    documents = SimpleDirectoryReader(
        input_files=[str(f) for f in new_files],
        file_extractor={
            ".pdf": PyMuPDFReader()
        }
    ).load_data()

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    save_track(updated_track)
    return index


def main():
    index = build_or_update_index()
    query_engine = index.as_query_engine()

    while True:
        q = input("\nQuestion> ").strip()
        if q in {"q", "quit", "exit"}:
            break
        res = query_engine.query(q)
        print("\nAnswer:\n", res)

        print("\nSources:")
        for i, node in enumerate(res.source_nodes):
            print(f"\n--- Source {i+1} ---")
            print(node.text[:300])  # 截断显示

if __name__ == "__main__":
    main()