"""Document resource storage and text-chunk/resource associations.

The FAISS index stores only searchable text and small metadata. Binary assets
are kept on disk and indexed through a JSON sidecar so the feature works
without adding a database dependency to the existing MVP.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import tempfile
import uuid
import zipfile
from dataclasses import asdict, dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote


class ResourceKind(str, Enum):
    IMAGE = "image"
    CHART = "chart"
    ATTACHMENT = "attachment"


@dataclass
class ParsedResource:
    """A binary resource extracted from an uploaded document."""

    filename: str
    mime_type: str
    content: bytes
    kind: ResourceKind = ResourceKind.IMAGE
    page_no: Optional[int] = None
    anchor_block_id: Optional[str] = None


@dataclass
class ResourceRecord:
    id: str
    document_id: str
    source_filename: str
    filename: str
    mime_type: str
    storage_key: str
    sha256: str
    kind: str
    page_no: Optional[int] = None
    anchor_block_id: Optional[str] = None
    alt_text: Optional[str] = None

    @classmethod
    def from_dict(cls, value: dict) -> "ResourceRecord":
        return cls(
            id=value["id"],
            document_id=value["document_id"],
            source_filename=value.get("source_filename", ""),
            filename=value.get("filename", "resource"),
            mime_type=value.get("mime_type", "application/octet-stream"),
            storage_key=value["storage_key"],
            sha256=value.get("sha256", ""),
            kind=value.get("kind", ResourceKind.ATTACHMENT.value),
            page_no=value.get("page_no"),
            anchor_block_id=value.get("anchor_block_id"),
            alt_text=value.get("alt_text"),
        )


def _safe_filename(filename: str) -> str:
    name = Path(filename or "resource").name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "resource"


def _unique(values: Iterable[str]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


class ResourceStore:
    """Store extracted files below a controlled root directory."""

    def __init__(self, root: str = "./rag_resources"):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, resource_id: str, filename: str, content: bytes) -> str:
        safe_name = _safe_filename(filename)
        relative_key = f"{resource_id}/{safe_name}"
        destination = (self.root / relative_key).resolve()
        if self.root not in destination.parents:
            raise ValueError("资源路径非法")

        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "wb") as file:
            file.write(content)
        return relative_key

    def path_for(self, storage_key: str) -> Path:
        path = (self.root / storage_key).resolve()
        if self.root not in path.parents:
            raise ValueError("资源路径非法")
        return path

    def delete(self, storage_key: str) -> None:
        path = self.path_for(storage_key)
        if path.exists() and path.is_file():
            path.unlink()
        if path.parent.exists() and path.parent != self.root:
            try:
                path.parent.rmdir()
            except OSError:
                pass

    def clear(self) -> None:
        for child in self.root.iterdir():
            if child.is_dir():
                for item in child.rglob("*"):
                    if item.is_file():
                        item.unlink()
                for directory in sorted(child.rglob("*"), reverse=True):
                    if directory.is_dir():
                        directory.rmdir()
                child.rmdir()
            elif child.is_file():
                child.unlink()


class ResourceRepository:
    """JSON sidecar repository for resource metadata and chunk links."""

    def __init__(self, metadata_path: str = "./vector_store/resources.json"):
        self.metadata_path = Path(metadata_path)
        self.links_path = self.metadata_path.with_name(
            f"{self.metadata_path.stem}_links{self.metadata_path.suffix}"
        )
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._resources: Dict[str, ResourceRecord] = self._load_resources()
        self._links: Dict[str, List[str]] = self._load_json(self.links_path, {})

    @staticmethod
    def _load_json(path: Path, default):
        if not path.exists():
            return default
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, ValueError):
            return default

    def _load_resources(self) -> Dict[str, ResourceRecord]:
        raw = self._load_json(self.metadata_path, {})
        if isinstance(raw, list):
            raw = {item.get("id"): item for item in raw if item.get("id")}
        return {
            resource_id: ResourceRecord.from_dict(value)
            for resource_id, value in raw.items()
            if isinstance(value, dict)
        }

    @staticmethod
    def _atomic_write(path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(value, file, ensure_ascii=False, indent=2)
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.remove(temporary_name)

    def _persist(self) -> None:
        self._atomic_write(
            self.metadata_path,
            {key: asdict(value) for key, value in self._resources.items()},
        )
        self._atomic_write(self.links_path, self._links)

    def save(self, resource: ResourceRecord) -> ResourceRecord:
        with self._lock:
            self._resources[resource.id] = resource
            self._persist()
        return resource

    def get(self, resource_id: str) -> Optional[ResourceRecord]:
        with self._lock:
            return self._resources.get(resource_id)

    def list_by_ids(self, resource_ids: Iterable[str]) -> List[ResourceRecord]:
        with self._lock:
            return [
                self._resources[resource_id]
                for resource_id in _unique(resource_ids)
                if resource_id in self._resources
            ]

    def list_by_chunk_ids(self, chunk_ids: Iterable[str]) -> Dict[str, List[ResourceRecord]]:
        with self._lock:
            return {
                chunk_id: self.list_by_ids(self._links.get(chunk_id, []))
                for chunk_id in _unique(chunk_ids)
            }

    def link_chunks(self, chunk_ids: Iterable[str], resource_ids: Iterable[str]) -> None:
        chunk_ids = _unique(chunk_ids)
        resource_ids = _unique(resource_ids)
        with self._lock:
            for chunk_id in chunk_ids:
                self._links[chunk_id] = resource_ids
            self._persist()

    def resources_for_source(self, source_filename: str) -> List[ResourceRecord]:
        with self._lock:
            return [
                resource
                for resource in self._resources.values()
                if resource.source_filename == source_filename
            ]

    def delete_source(self, source_filename: str) -> List[ResourceRecord]:
        with self._lock:
            resources = self.resources_for_source(source_filename)
            resource_ids = {resource.id for resource in resources}
            self._resources = {
                key: value
                for key, value in self._resources.items()
                if key not in resource_ids
            }
            self._links = {
                chunk_id: [item for item in ids if item not in resource_ids]
                for chunk_id, ids in self._links.items()
                if any(item not in resource_ids for item in ids)
            }
            self._persist()
            return resources

    def clear(self) -> None:
        with self._lock:
            self._resources = {}
            self._links = {}
            self._persist()

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_resources": len(self._resources),
                "linked_chunks": len(self._links),
            }


def _guess_mime_type(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def _extract_zip_resources(
    content: bytes,
    filename: str,
) -> List[ParsedResource]:
    extension = Path(filename).suffix.lower()
    media_prefixes = {
        ".docx": "word/media/",
        ".pptx": "ppt/media/",
        ".xlsx": "xl/media/",
    }
    embedding_prefixes = {
        ".docx": "word/embeddings/",
        ".pptx": "ppt/embeddings/",
        ".xlsx": "xl/embeddings/",
    }
    if extension not in media_prefixes:
        return []

    resources = []
    with zipfile.ZipFile(BytesIO(content)) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            if item.filename.startswith(media_prefixes[extension]):
                item_name = Path(item.filename).name
                resources.append(
                    ParsedResource(
                        filename=item_name,
                        mime_type=_guess_mime_type(item_name),
                        content=archive.read(item),
                        kind=ResourceKind.IMAGE,
                    )
                )
            elif item.filename.startswith(embedding_prefixes[extension]):
                item_name = Path(item.filename).name
                resources.append(
                    ParsedResource(
                        filename=item_name,
                        mime_type=_guess_mime_type(item_name),
                        content=archive.read(item),
                        kind=ResourceKind.ATTACHMENT,
                    )
                )
    return resources


def _extract_pdf_resources(content: bytes) -> List[ParsedResource]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return []

    resources = []
    try:
        reader = PdfReader(BytesIO(content))
        for page_no, page in enumerate(reader.pages, 1):
            try:
                images = page.images
            except (AttributeError, TypeError, ValueError):
                images = []

            for image in images:
                image_name = getattr(image, "name", f"page_{page_no}_image")
                image_content = getattr(image, "data", None)
                if image_content:
                    resources.append(
                        ParsedResource(
                            filename=image_name,
                            mime_type=_guess_mime_type(image_name),
                            content=image_content,
                            kind=ResourceKind.IMAGE,
                            page_no=page_no,
                        )
                    )

        attachments = getattr(reader, "attachments", {}) or {}
        for attachment_name, attachment_content in attachments.items():
            if isinstance(attachment_content, list):
                attachment_content = attachment_content[0] if attachment_content else b""
            if isinstance(attachment_content, str):
                attachment_content = attachment_content.encode("utf-8")
            if attachment_content:
                resources.append(
                    ParsedResource(
                        filename=attachment_name,
                        mime_type=_guess_mime_type(attachment_name),
                        content=attachment_content,
                        kind=ResourceKind.ATTACHMENT,
                    )
                )
    except Exception:
        # Text ingestion should continue even when optional asset extraction
        # is not supported by a particular PDF implementation.
        return resources

    return resources


def extract_resources(content: bytes, filename: str) -> List[ParsedResource]:
    """Extract embedded images and attachments from supported documents."""
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        return _extract_pdf_resources(content)
    if extension in {".docx", ".pptx", ".xlsx"}:
        try:
            return _extract_zip_resources(content, filename)
        except (OSError, zipfile.BadZipFile):
            return []
    return []


class ResourceManager:
    """Coordinates extraction, persistence, file access and rendering."""

    def __init__(
        self,
        root: str = "./rag_resources",
        metadata_path: str = "./vector_store/resources.json",
        public_url_prefix: str = "/api/rag/resource",
    ):
        self.store = ResourceStore(root)
        self.repository = ResourceRepository(metadata_path)
        self.public_url_prefix = public_url_prefix.rstrip("/")

    def ingest(
        self,
        content: bytes,
        filename: str,
        document_id: Optional[str] = None,
    ) -> List[ResourceRecord]:
        document_id = document_id or uuid.uuid4().hex
        parsed_resources = extract_resources(content, filename)
        records = []
        for parsed in parsed_resources:
            resource_id = uuid.uuid4().hex
            storage_key = self.store.put(
                resource_id=resource_id,
                filename=parsed.filename,
                content=parsed.content,
            )
            record = ResourceRecord(
                id=resource_id,
                document_id=document_id,
                source_filename=filename,
                filename=parsed.filename,
                mime_type=parsed.mime_type,
                storage_key=storage_key,
                sha256=hashlib.sha256(parsed.content).hexdigest(),
                kind=parsed.kind.value,
                page_no=parsed.page_no,
                anchor_block_id=parsed.anchor_block_id,
            )
            self.repository.save(record)
            records.append(record)
        return records

    def to_ref(self, resource: ResourceRecord) -> dict:
        url = f"{self.public_url_prefix}/{quote(resource.id, safe='')}"
        label = resource.filename.replace("]", "\\]")
        if resource.kind in {
            ResourceKind.IMAGE.value,
            ResourceKind.CHART.value,
        }:
            markdown = f"![{label}]({url})"
        else:
            markdown = f"[下载 {label}]({url})"
        return {
            "id": resource.id,
            "kind": resource.kind,
            "filename": resource.filename,
            "mime_type": resource.mime_type,
            "page_no": resource.page_no,
            "url": url,
            "markdown": markdown,
        }

    def enrich_search_results(self, results: List[dict]) -> List[dict]:
        chunk_ids = [
            item.get("metadata", {}).get("chunk_id")
            for item in results
        ]
        links = self.repository.list_by_chunk_ids(chunk_ids)
        enriched = []
        for item in results:
            metadata = item.get("metadata", {})
            resources = links.get(metadata.get("chunk_id"), [])
            if not resources:
                resources = self.repository.list_by_ids(
                    metadata.get("resource_ids", [])
                )
            enriched_item = dict(item)
            enriched_item["resources"] = [self.to_ref(resource) for resource in resources]
            enriched.append(enriched_item)
        return enriched

    def collect_refs(self, results: List[dict]) -> List[dict]:
        refs = {}
        for item in results:
            for resource in item.get("resources", []):
                refs[resource["id"]] = resource
        return list(refs.values())

    @staticmethod
    def resolve_asset_tags(answer: str, resources: List[dict]) -> str:
        """Replace only resource IDs returned by the current retrieval."""
        allowed = {resource["id"]: resource for resource in resources}

        def replace(match):
            resource = allowed.get(match.group(1))
            return resource.get("markdown", "") if resource else ""

        return re.sub(r"\[\[asset:([A-Za-z0-9_-]+)\]\]", replace, answer)

    def path_for(self, resource_id: str) -> Optional[ResourceRecord]:
        resource = self.repository.get(resource_id)
        if resource is None:
            return None
        path = self.store.path_for(resource.storage_key)
        if not path.exists() or not path.is_file():
            return None
        return resource

    def file_path(self, resource: ResourceRecord) -> Path:
        return self.store.path_for(resource.storage_key)

    def delete_source(self, source_filename: str) -> int:
        resources = self.repository.delete_source(source_filename)
        for resource in resources:
            self.store.delete(resource.storage_key)
        return len(resources)

    def clear(self) -> None:
        self.repository.clear()
        self.store.clear()

    def stats(self) -> dict:
        return self.repository.stats()
