"""Backend-independent assets, geometry and source relationships."""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Literal

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IndexArray = NDArray[np.int64]
type NumericArray = NDArray[np.int8] | NDArray[np.uint8] | NDArray[np.int16] | NDArray[np.uint16] | NDArray[np.uint32] | NDArray[np.float32]
type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type AssetFormat = Literal["glb", "stl"]
type TextureSlot = Literal["base_color", "metallic_roughness", "normal", "occlusion", "emissive"]


@dataclass(frozen=True)
class AssetRef:
    source_id: str
    asset_id: str
    relative_path: str
    expected_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("source_id", "asset_id", "relative_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if self.expected_sha256 is not None:
            if not isinstance(self.expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", self.expected_sha256):
                raise ValueError("expected_sha256 must contain 64 hexadecimal characters")
            object.__setattr__(self, "expected_sha256", self.expected_sha256.lower())


@dataclass
class AssetSource:
    ref: AssetRef
    format: AssetFormat
    resolved_path: Path
    actual_sha256: str
    selected_scene_index: int | None = None
    stl_header: bytes | None = None
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class AttributeSource:
    accessor_index: int
    component_type: int
    normalized: bool


@dataclass
class MeshData:
    vertices: FloatArray
    faces: IndexArray
    vertex_normals: FloatArray | None = None
    face_normals: FloatArray | None = None
    uv_sets: dict[int, FloatArray] = field(default_factory=dict)
    color_sets: dict[int, FloatArray] = field(default_factory=dict)
    tangents: FloatArray | None = None
    material_index: int | None = None
    custom_attributes: dict[str, NumericArray] = field(default_factory=dict)
    attribute_sources: dict[str, AttributeSource] = field(default_factory=dict)
    source_mesh_index: int | None = None
    source_primitive_index: int | None = None


@dataclass
class NodeData:
    parent_index: int | None
    mesh_indices: list[int]
    local_transform: FloatArray
    source_node_index: int | None = None


@dataclass(frozen=True)
class TextureTransform:
    offset: tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    scale: tuple[float, float] = (1.0, 1.0)
    texcoord_override: int | None = None


@dataclass(frozen=True)
class TextureBinding:
    texture_index: int
    texcoord: int = 0
    transform: TextureTransform | None = None


@dataclass
class MaterialData:
    base_color_factor: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic_factor: float = 1.0
    roughness_factor: float = 1.0
    emissive_factor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    alpha_mode: Literal["OPAQUE", "MASK", "BLEND"] = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    unlit: bool = False
    textures: dict[TextureSlot, TextureBinding] = field(default_factory=dict)
    normal_scale: float = 1.0
    occlusion_strength: float = 1.0
    source_material_index: int | None = None


@dataclass(frozen=True)
class ImageData:
    data: bytes
    mime_type: Literal["image/png", "image/jpeg"]
    source_image_index: int


@dataclass(frozen=True)
class TextureData:
    image_index: int
    sampler_index: int | None
    source_texture_index: int


@dataclass(frozen=True)
class SamplerData:
    mag_filter: int | None = None
    min_filter: int | None = None
    wrap_s: int = 10497
    wrap_t: int = 10497


@dataclass
class Asset:
    source: AssetSource
    meshes: list[MeshData]
    nodes: list[NodeData]
    materials: list[MaterialData] = field(default_factory=list)
    images: list[ImageData] = field(default_factory=list)
    textures: list[TextureData] = field(default_factory=list)
    samplers: list[SamplerData] = field(default_factory=list)
