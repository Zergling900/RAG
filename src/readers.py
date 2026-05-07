from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.readers.file import PyMuPDFReader


CODE_EXTS = {".py", ".cpp", ".h", ".hpp", ".c", ".cc", ".inc", ".sh"}
TEXT_EXTS = {".md", ".txt", ".p1", ".p2", ".p3", ".p4", ".p5", ".pr2", ".json"}
SKIP_PARTS = {".git", "__pycache__", ".venv"}
MAX_TEXT_CHARS = 24000
LINE_CHUNK_SIZE = 120
LINE_CHUNK_OVERLAP = 12
@dataclass(frozen=True)
class SourceFile:
    path: Path
    track_key: str
    metadata: dict[str, Any]


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"projects": []}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data.setdefault("projects", [])
    return data


def is_probably_text(path: Path, sample_size: int = 4096) -> bool:
    try:
        sample = path.read_bytes()[:sample_size]
    except OSError:
        return False
    if not sample:
        return True
    return b"\x00" not in sample


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, Path):
            normalized[key] = str(value)
        elif isinstance(value, (list, tuple, dict)):
            normalized[key] = json.dumps(value, ensure_ascii=False)
        else:
            normalized[key] = value
    return normalized


def compact_related_papers(paths: Iterable[str]) -> dict[str, Any]:
    papers = sorted(paths)
    return {"related_paper_count": len(papers)}


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "untitled"


def paper_metadata(path: Path, data_dir: Path) -> dict[str, Any]:
    rel = path.relative_to(data_dir).as_posix()
    parent_parts = path.relative_to(data_dir).parts[:-1]
    return normalize_metadata(
        {
            "source_type": "paper",
            "paper_id": slugify(path.stem[:120]),
            "title": path.stem,
            "file_path": str(path),
            "relative_path": rel,
            "library_path": "/".join(parent_parts),
            "library_section": parent_parts[0] if parent_parts else "",
            "topic_path": " / ".join(parent_parts[1:]) if len(parent_parts) > 1 else "",
            "file_name": path.name,
        }
    )


def resolve_related_papers(globs: Iterable[str], project_root: Path) -> list[str]:
    resolved: set[str] = set()
    for pattern in globs:
        pattern_path = Path(pattern)
        search_root = pattern_path if pattern_path.is_absolute() else project_root / pattern
        for path in project_root.glob(
            str(search_root.relative_to(project_root))
            if search_root.is_relative_to(project_root)
            else pattern
        ):
            if path.is_file() and path.suffix.lower() == ".pdf":
                resolved.add(str(path))
    return sorted(resolved)


def resolve_source_root(source_root: str, project_root: Path) -> Path:
    path = Path(source_root).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def iter_manifest_sources(manifest_path: Path, project_root: Path) -> list[SourceFile]:
    manifest = load_manifest(manifest_path)
    sources: dict[str, SourceFile] = {}

    for project in manifest.get("projects", []):
        source_root = resolve_source_root(project["source_root"], project_root)
        if not source_root.exists():
            continue

        project_meta = {
            "project_id": project.get("project_id"),
            "project_title": project.get("title"),
        }

        for component in project.get("components", []):
            related_papers = resolve_related_papers(
                component.get("related_paper_globs", []), project_root
            )
            related_paper_meta = compact_related_papers(related_papers)
            component_meta = {
                **project_meta,
                "component_id": component.get("component_id"),
                "component_title": component.get("title"),
                "source_role": component.get("source_role"),
                "system": component.get("system"),
                "method": component.get("method"),
                **related_paper_meta,
            }

            for pattern in component.get("include", []):
                for path in sorted(source_root.glob(pattern)):
                    if not path.is_file():
                        continue
                    if any(part in SKIP_PARTS for part in path.parts):
                        continue
                    if path.suffix.lower() not in CODE_EXTS | TEXT_EXTS:
                        continue
                    if not is_probably_text(path):
                        continue

                    rel = path.relative_to(source_root).as_posix()
                    key = f"manifest:{project.get('project_id')}:{rel}"
                    metadata = normalize_metadata(
                        {
                            **component_meta,
                            "source_type": infer_source_type(path),
                            "file_path": str(path),
                            "relative_path": rel,
                            "file_name": path.name,
                            "extension": path.suffix.lower(),
                        }
                    )
                    sources[key] = SourceFile(path=path, track_key=key, metadata=metadata)

    return sorted(sources.values(), key=lambda item: item.track_key)


def iter_data_dir_sources(data_dir: Path) -> list[SourceFile]:
    sources = []
    if not data_dir.exists():
        return sources

    for path in sorted(file for file in data_dir.rglob("*") if file.is_file()):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        suffix = path.suffix.lower()
        if suffix not in {".pdf"} | CODE_EXTS | TEXT_EXTS:
            continue
        if suffix != ".pdf" and not is_probably_text(path):
            continue

        rel = path.relative_to(data_dir).as_posix()
        metadata = paper_metadata(path, data_dir) if suffix == ".pdf" else normalize_metadata(
            {
                "source_type": infer_source_type(path),
                "file_path": str(path),
                "relative_path": rel,
                "file_name": path.name,
                "extension": suffix,
            }
        )
        sources.append(SourceFile(path=path, track_key=rel, metadata=metadata))

    return sources


def iter_sources(data_dir: Path, manifest_path: Path, project_root: Path) -> list[SourceFile]:
    manifest_sources = iter_manifest_sources(manifest_path, project_root)
    manifest_paths = {source.path.resolve() for source in manifest_sources}
    sources = {
        source.track_key: source
        for source in iter_data_dir_sources(data_dir)
        if source.path.resolve() not in manifest_paths
    }
    for source in manifest_sources:
        sources[source.track_key] = source
    return sorted(sources.values(), key=lambda item: item.track_key)


def infer_source_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "paper"
    if suffix in CODE_EXTS:
        return "code"
    if suffix in {".p1", ".p2", ".p3", ".p4", ".p5", ".pr2"}:
        return "parameter"
    if suffix in {".md", ".txt"}:
        return "note"
    return "text"


def load_source_documents(sources: Iterable[SourceFile]) -> list[Document]:
    documents: list[Document] = []
    for source in sources:
        suffix = source.path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(load_pdf(source))
        elif suffix == ".py":
            documents.extend(load_python_code(source))
        elif suffix in CODE_EXTS:
            documents.extend(load_line_chunked_code(source))
        else:
            documents.extend(load_text_chunks(source))
    return documents


def load_pdf(source: SourceFile) -> list[Document]:
    return SimpleDirectoryReader(
        input_files=[str(source.path)],
        file_metadata=lambda _: source.metadata,
        file_extractor={".pdf": PyMuPDFReader()},
    ).load_data()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def make_document(text: str, metadata: dict[str, Any]) -> Document:
    return Document(text=text, metadata=normalize_metadata(metadata))


def load_python_code(source: SourceFile) -> list[Document]:
    text = read_text(source.path)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return load_text_chunks(source)

    lines = text.splitlines()
    docs = []
    module_doc = ast.get_docstring(tree)
    imports = [
        line.strip()
        for line in lines
        if line.lstrip().startswith(("import ", "from "))
    ][:40]

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start)
        chunk = "\n".join(lines[start - 1:end])
        symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"
        header = format_code_header(source, symbol_type, node.name, start, end, module_doc, imports)
        docs.append(
            make_document(
                f"{header}\n\n```python\n{chunk}\n```",
                {
                    **source.metadata,
                    "symbol_type": symbol_type,
                    "symbol_name": node.name,
                    "line_start": start,
                    "line_end": end,
                },
            )
        )

    if docs:
        return docs
    return load_text_chunks(source)


def format_code_header(
    source: SourceFile,
    symbol_type: str,
    symbol_name: str,
    start: int,
    end: int,
    module_doc: str | None,
    imports: list[str],
) -> str:
    parts = [
        f"Project: {source.metadata.get('project_title', source.metadata.get('project_id', ''))}",
        f"Component: {source.metadata.get('component_title', source.metadata.get('component_id', ''))}",
        f"Source role: {source.metadata.get('source_role', '')}",
        f"File: {source.metadata.get('relative_path', source.path.name)}",
        f"Symbol: {symbol_type} {symbol_name}",
        f"Lines: {start}-{end}",
    ]
    if module_doc:
        parts.append(f"Module docstring: {module_doc[:800]}")
    if imports:
        parts.append("Imports:\n" + "\n".join(imports))
    return "\n".join(part for part in parts if part)


def load_line_chunked_code(source: SourceFile) -> list[Document]:
    text = read_text(source.path)
    lines = text.splitlines()
    docs = []
    index = 0
    start = 1
    while start <= len(lines):
        end = min(len(lines), start + LINE_CHUNK_SIZE - 1)
        chunk = "\n".join(lines[start - 1:end])
        symbol_name = infer_nearest_symbol(lines, start)
        language = source.path.suffix.lower().lstrip(".") or "text"
        header = "\n".join(
            [
                f"Project: {source.metadata.get('project_title', source.metadata.get('project_id', ''))}",
                f"Component: {source.metadata.get('component_title', source.metadata.get('component_id', ''))}",
                f"Source role: {source.metadata.get('source_role', '')}",
                f"File: {source.metadata.get('relative_path', source.path.name)}",
                f"Lines: {start}-{end}",
                f"Nearest symbol: {symbol_name}",
            ]
        )
        docs.append(
            make_document(
                f"{header}\n\n```{language}\n{chunk}\n```",
                {
                    **source.metadata,
                    "symbol_type": "code_chunk",
                    "symbol_name": symbol_name,
                    "chunk_index": index,
                    "line_start": start,
                    "line_end": end,
                },
            )
        )
        index += 1
        if end == len(lines):
            break
        start = max(end - LINE_CHUNK_OVERLAP + 1, start + 1)
    return docs


def infer_nearest_symbol(lines: list[str], start_line: int) -> str:
    pattern = re.compile(
        r"^\s*(?:class|struct|void|int|double|float|bool|auto|std::[\w:<>]+|[\w:<>]+)\s+([A-Za-z_]\w*)\s*(?:\(|:|\{)"
    )
    for line in reversed(lines[:start_line]):
        match = pattern.match(line)
        if match:
            return match.group(1)
    return "module"


def load_text_chunks(source: SourceFile) -> list[Document]:
    text = read_text(source.path)
    if len(text) <= MAX_TEXT_CHARS:
        return [
            make_document(
                format_text_document(source, text),
                {**source.metadata, "chunk_index": 0},
            )
        ]

    docs = []
    lines = text.splitlines()
    index = 0
    start = 1
    while start <= len(lines):
        end = min(len(lines), start + LINE_CHUNK_SIZE - 1)
        chunk = "\n".join(lines[start - 1:end])
        docs.append(
            make_document(
                format_text_document(source, chunk),
                {
                    **source.metadata,
                    "chunk_index": index,
                    "line_start": start,
                    "line_end": end,
                },
            )
        )
        index += 1
        if end == len(lines):
            break
        start = max(end - LINE_CHUNK_OVERLAP + 1, start + 1)
    return docs


def format_text_document(source: SourceFile, text: str) -> str:
    return "\n".join(
        [
            f"Source file: {source.metadata.get('relative_path', source.path.name)}",
            f"Source type: {source.metadata.get('source_type', '')}",
            f"Component: {source.metadata.get('component_title', source.metadata.get('component_id', ''))}",
            "",
            text,
        ]
    )
