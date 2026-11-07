"""Import failures and small serializable import results."""

from dataclasses import dataclass, field, asdict
from typing import Literal, cast

from meshengineer.core.assets import AssetFormat, AssetRef, JsonValue

type ImportErrorCode = Literal["invalid_path", "read_error", "hash_mismatch", "invalid_asset", "unsupported_asset"]


class AssetImportError(Exception):
    def __init__(self, ref: AssetRef, code: ImportErrorCode, message: str,
                 actual_sha256: str | None = None) -> None:
        super().__init__(message)
        self.ref = ref
        self.code = code
        self.message = message
        self.actual_sha256 = actual_sha256


@dataclass
class ImportRecord:
    ref: AssetRef
    status: Literal["ok", "failed"]
    source_path: str
    actual_sha256: str | None = None
    format: AssetFormat | None = None
    mesh_count: int | None = None
    vertex_count: int | None = None
    face_count: int | None = None
    instance_count: int | None = None
    selected_scene_index: int | None = None
    code: ImportErrorCode | None = None
    message: str | None = None
    notes: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], asdict(self))


@dataclass
class BatchSummary:
    total: int
    report_path: str
    processed: int = 0
    ok: int = 0
    failed: int = 0
    reported: int = 0
    completed: bool = False
    abort_reason: str | None = None

    def to_json(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], asdict(self))
