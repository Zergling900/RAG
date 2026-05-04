import json
from pathlib import Path
from hashlib import md5

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore

from src.config import (
    PROJECT_ROOT,
    MILVUS_URI,
    MILVUS_COLLECTION,
    EMBED_DIM,
    DATA_DIR,
    MANIFEST_PATH,
)
from src.readers import iter_sources, load_source_documents


TRACK_FILE = Path("ingested_files.json")


def file_hash(path: Path):
    return md5(path.read_bytes()).hexdigest()


def load_track():
    if TRACK_FILE.exists():
        raw = TRACK_FILE.read_text().strip()
        if raw:
            return json.loads(raw)
    return {}


def save_track(data):
    TRACK_FILE.write_text(json.dumps(data, indent=2))


def get_new_documents():
    tracked = load_track()
    new_files = []
    updated_track = {}

    for source in iter_sources(DATA_DIR, MANIFEST_PATH, PROJECT_ROOT):
        h = file_hash(source.path)
        key = source.track_key
        updated_track[key] = h

        if key not in tracked or tracked[key] != h:
            new_files.append(source)

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

    documents = load_source_documents(new_files)
    print(f"Loaded {len(documents)} document chunks.")

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
            metadata = node.node.metadata
            print("file:", metadata.get("file_path"))
            print("type:", metadata.get("source_type"))
            print("component:", metadata.get("component_id"))
            print(node.text[:300])  # 截断显示

if __name__ == "__main__":
    main()
