"""Validated GLB adaptation; native backend objects stay in this module."""

import base64
import binascii
import math
import re
import struct
from typing import Literal, Never, cast
from urllib.parse import unquote_to_bytes

import numpy as np
from numpy.typing import NDArray
from pygltflib import GLTF2  # type: ignore[import-untyped]

from meshengineer.core.assets import (
    Asset, AssetSource, AttributeSource, FloatArray, ImageData, JsonValue, MaterialData,
    MeshData, NodeData, NumericArray, SamplerData, TextureBinding, TextureData, TextureSlot, TextureTransform,
)
from meshengineer.core.imports import AssetImportError, ImportErrorCode
from meshengineer.io.loader import parse_json

type Object = dict[str, JsonValue]
_COMPRESSION = {"KHR_draco_mesh_compression", "EXT_meshopt_compression", "KHR_meshopt_compression"}
_SUPPORTED = _COMPRESSION | {"KHR_texture_transform", "KHR_materials_unlit"}
type Component = np.int8 | np.uint8 | np.int16 | np.uint16 | np.uint32 | np.float32
_DTYPES: dict[int, np.dtype[Component]] = {5120: np.dtype("i1"), 5121: np.dtype("u1"), 5122: np.dtype("<i2"),
           5123: np.dtype("<u2"), 5125: np.dtype("<u4"), 5126: np.dtype("<f4")}
_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


class _Reader:
    def __init__(self, raw: bytes, source: AssetSource) -> None:
        self.source = source
        if len(raw) < 12 or raw[:4] != b"glTF": self.fail("Invalid GLB header")
        version, length = struct.unpack_from("<II", raw, 4)
        if version != 2: self.fail(f"Unsupported GLB version {version}", "unsupported_asset")
        if length != len(raw): self.fail("GLB declared length does not match input")
        chunks: list[tuple[int, bytes]] = []
        offset = 12
        while offset < len(raw):
            if offset + 8 > len(raw): self.fail("Truncated GLB chunk header")
            size, kind = struct.unpack_from("<II", raw, offset)
            offset += 8
            if size % 4 or offset + size > len(raw): self.fail("Invalid GLB chunk length or alignment")
            chunks.append((kind, raw[offset:offset+size]))
            offset += size
        if not chunks or chunks[0][0] != 0x4e4f534a: self.fail("First GLB chunk must be JSON")
        if sum(k == 0x4e4f534a for k, _ in chunks) != 1: self.fail("GLB must contain one JSON chunk")
        bins = [i for i,(kind,_) in enumerate(chunks) if kind == 0x004e4942]
        if len(bins)>1 or (bins and bins[0]!=1): self.fail("BIN must be the second and only binary chunk")
        try:
            self.doc = self.obj(parse_json(chunks[0][1].decode("utf-8"), "GLB JSON"), "root")
        except (ValueError, UnicodeError) as error:
            self.fail(f"Invalid GLB JSON: {error}")
        asset = self.obj(self.doc.get("asset"), "asset")
        for name in ("version", "minVersion"):
            value = asset.get(name)
            if value is None and name == "minVersion": continue
            if not isinstance(value,str) or not re.fullmatch(r"\d+\.\d+",value): self.fail(f"Invalid asset.{name}")
            parts = tuple(int(v) for v in value.split("."))
            if (name == "version" and parts != (2,0)) or (name == "minVersion" and parts > (2,0)):
                self.fail(f"Unsupported asset.{name}: {value}", "unsupported_asset")
        self.tables = {name:self.table(name) for name in ("nodes","meshes","accessors","bufferViews","buffers",
                                                       "materials","images","textures","samplers","scenes","cameras")}
        used = self.names(self.doc.get("extensionsUsed",[]),"extensionsUsed")
        required = self.names(self.doc.get("extensionsRequired",[]),"extensionsRequired")
        if not required <= used: self.fail("extensionsRequired must be a subset of extensionsUsed")
        if used - _SUPPORTED or required & _COMPRESSION:
            self.fail(f"Unsupported GLB extensions: {sorted((used-_SUPPORTED)|(required&_COMPRESSION))}","unsupported_asset")
        self.scan_extensions(self.doc,used)
        if self.arr(self.doc.get("animations",[]),"animations"):
            self.fail("Animations are not supported","unsupported_asset")
        for node in self.tables["nodes"]:
            if "skin" in node: self.fail("Applied skins are not supported","unsupported_asset")
        for mesh in self.tables["meshes"]:
            for primitive in self.arr(mesh.get("primitives"),"mesh.primitives"):
                if self.arr(self.obj(primitive,"primitive").get("targets",[]),"morph targets"):
                    self.fail("Morph targets are not supported","unsupported_asset")
        # Filter legal unknown chunks before pygltflib, which decodes chunk IDs as text.
        known = [(k,b) for k,b in chunks if k in (0x4e4f534a,0x004e4942)]
        backend_raw = raw
        if len(known) != len(chunks):
            body=b"".join(struct.pack("<II",len(b),k)+b for k,b in known)
            backend_raw=struct.pack("<4sII",b"glTF",2,12+len(body))+body
        try:
            native = GLTF2.load_from_bytes(backend_raw)
            binary = native.binary_blob()
        except (ValueError, TypeError, KeyError, IndexError, OverflowError, struct.error) as error:
            self.fail(f"Invalid GLB structure: {error}")
        if binary is not None and not isinstance(binary,bytes): self.fail("Invalid native binary buffer")
        self.buffers: list[bytes | None] = []
        for i,buffer in enumerate(self.tables["buffers"]):
            size=self.integer(buffer.get("byteLength"),"buffer.byteLength",1)
            if "uri" in buffer:
                payload,_=self.embedded(buffer["uri"])
                if len(payload)!=size: self.fail("Embedded buffer length mismatch")
            elif i==0 and binary is not None:
                payload=binary
                if not size<=len(payload)<=size+3: self.fail("BIN buffer length mismatch")
                payload=payload[:size]
            else:
                extensions=self.obj(buffer.get("extensions",{}),"buffer.extensions")
                if any(self.obj(v,"buffer compression").get("fallback") is True for k,v in extensions.items() if k in _COMPRESSION):
                    self.buffers.append(None)
                    continue
                self.fail("Buffer has no embedded payload")
            self.buffers.append(payload)
        # All resource URIs must be self contained, including unused images.
        for image in self.tables["images"]:
            if "uri" in image: self.embedded(image["uri"])

    def fail(self, message: str, code: ImportErrorCode = "invalid_asset") -> Never:
        raise AssetImportError(self.source.ref,code,message,self.source.actual_sha256)

    def obj(self, value: JsonValue, label: str) -> Object:
        if not isinstance(value,dict): self.fail(f"{label} must be an object")
        return value

    def arr(self, value: JsonValue, label: str) -> list[JsonValue]:
        if not isinstance(value,list): self.fail(f"{label} must be an array")
        return value

    def integer(self, value: JsonValue, label: str, minimum: int = 0) -> int:
        if not isinstance(value,int) or isinstance(value,bool) or value<minimum: self.fail(f"Invalid {label}")
        return value

    def index(self, value: JsonValue, table: str) -> int:
        index=self.integer(value,f"{table} index")
        if index>=len(self.tables[table]): self.fail(f"{table} index {index} is out of range")
        return index

    def table(self, name: str) -> list[Object]:
        return [self.obj(v,name) for v in self.arr(self.doc.get(name,[]),name)]

    def names(self, value: JsonValue, label: str) -> set[str]:
        items=self.arr(value,label)
        if any(not isinstance(v,str) for v in items): self.fail(f"Invalid {label}")
        return {v for v in items if isinstance(v,str)}

    def scan_extensions(self, value: JsonValue, used: set[str]) -> None:
        if isinstance(value,dict):
            for key,item in value.items():
                if key=="extras": continue
                if key=="extensions":
                    extensions=self.obj(item,"extensions")
                    if extensions.keys()-used: self.fail("Extension is missing from extensionsUsed")
                    if extensions.keys()-_SUPPORTED: self.fail("Unsupported extension","unsupported_asset")
                self.scan_extensions(item,used)
        elif isinstance(value,list):
            for item in value: self.scan_extensions(item,used)

    def embedded(self, value: JsonValue) -> tuple[bytes,str]:
        if not isinstance(value,str): self.fail("Resource URI must be a string")
        if not value.startswith("data:"): self.fail("External resource URIs are not supported","unsupported_asset")
        header,separator,payload=value[5:].partition(",")
        if not separator: self.fail("Malformed data URI")
        try:
            data=base64.b64decode(payload,validate=True) if header.endswith(";base64") else unquote_to_bytes(payload)
        except (ValueError,binascii.Error): self.fail("Malformed data URI payload")
        return data,header.split(";",1)[0]

    def view(self, index: int) -> tuple[bytes,int,int,Object]:
        view=self.tables["bufferViews"][index]
        buffer_index=self.index(view.get("buffer"),"buffers")
        payload=self.buffers[buffer_index]
        if payload is None: self.fail("Compression has no actual uncompressed fallback","unsupported_asset")
        start=self.integer(view.get("byteOffset",0),"bufferView.byteOffset")
        size=self.integer(view.get("byteLength"),"bufferView.byteLength",1)
        if start+size>len(payload): self.fail("bufferView exceeds its buffer")
        extensions=self.obj(view.get("extensions",{}),"bufferView.extensions")
        for name in extensions:
            if name in _COMPRESSION:
                note=f"Using uncompressed fallback for {name}; compressed equivalence is not verified."
                if note not in self.source.notes: self.source.notes.append(note)
        return payload,start,size,view

    def accessor(self, index: int, vertex: bool) -> tuple[NumericArray,AttributeSource,str]:
        accessor=self.tables["accessors"][index]
        if "sparse" in accessor: self.fail("Sparse accessors are not supported","unsupported_asset")
        count=self.integer(accessor.get("count"),"accessor.count",1)
        kind=accessor.get("type")
        if not isinstance(kind,str) or kind not in _WIDTHS: self.fail("Unsupported accessor shape","unsupported_asset")
        component=self.integer(accessor.get("componentType"),"accessor.componentType")
        if component not in _DTYPES: self.fail("Invalid accessor componentType")
        normalized=accessor.get("normalized",False)
        if not isinstance(normalized,bool) or (normalized and component in (5125,5126)): self.fail("Invalid normalized accessor")
        if vertex and component==5125: self.fail("UNSIGNED_INT is only allowed for indices")
        dtype=_DTYPES[component];width=_WIDTHS[kind];element=dtype.itemsize*width
        relative=self.integer(accessor.get("byteOffset",0),"accessor.byteOffset")
        data: NDArray[Component]
        if "bufferView" not in accessor:
            if relative: self.fail("Accessor without bufferView cannot have a nonzero offset")
            data=np.zeros((count,width),dtype=dtype)
        else:
            view_index=self.index(accessor["bufferView"],"bufferViews")
            payload,start,size,view=self.view(view_index)
            stride=self.integer(view.get("byteStride",element),"bufferView.byteStride",1)
            if "byteStride" in view and (stride<4 or stride>252 or stride%4):
                self.fail("Declared byteStride must be a multiple of four from 4 to 252")
            if stride<element or stride%dtype.itemsize or relative%dtype.itemsize or (start+relative)%dtype.itemsize:
                self.fail("Invalid accessor alignment or stride")
            if vertex and (relative%4 or (start+relative)%4 or stride%4): self.fail("Vertex attributes require 4-byte alignment")
            if not vertex and "byteStride" in view: self.fail("Index bufferViews cannot have byteStride")
            if relative+(count-1)*stride+element>size: self.fail("Accessor exceeds its bufferView")
            data=np.ndarray((count,width),dtype=dtype,buffer=payload,offset=start+relative,strides=(stride,dtype.itemsize)).copy()
        if not np.isfinite(data).all(): self.fail(f"Non-finite accessor {index}")
        return cast(NumericArray,data),AttributeSource(index,component,normalized),kind

    def floating(self, data: NumericArray, source: AttributeSource) -> FloatArray:
        result=data.astype(np.float64)
        if source.normalized:
            maximum=float({5120:127,5121:255,5122:32767,5123:65535}[source.component_type])
            result/=maximum
            if data.dtype.kind=="i": result=np.maximum(result,-1.0)
        return result

    def geometry(self, mesh_index: int, primitive_index: int, primitive: Object) -> MeshData:
        if primitive.get("mode",4)!=4: self.fail("Only TRIANGLES primitives are supported","unsupported_asset")
        attributes=self.obj(primitive.get("attributes"),"primitive.attributes")
        if "POSITION" not in attributes: self.fail("Primitive is missing POSITION")
        extensions=self.obj(primitive.get("extensions",{}),"primitive.extensions")
        compressed=extensions.keys()&_COMPRESSION
        if compressed:
            for semantic,value in attributes.items():
                accessor=self.tables["accessors"][self.index(value,"accessors")]
                if "bufferView" not in accessor:
                    self.fail(f"{semantic} has only compression metadata, no ordinary fallback","unsupported_asset")
            if "indices" in primitive and "bufferView" not in self.tables["accessors"][self.index(primitive["indices"],"accessors")]:
                self.fail("Indices have no ordinary fallback","unsupported_asset")
            for name in compressed:
                self.source.notes.append(f"Using uncompressed fallback for {name}; compressed equivalence is not verified.")
        position_index=self.index(attributes["POSITION"],"accessors")
        position,origin,kind=self.accessor(position_index,True)
        if kind!="VEC3" or origin.component_type!=5126: self.fail("POSITION must be FLOAT VEC3")
        vertices=self.floating(position,origin)
        if "indices" in primitive:
            indices,index_origin,index_kind=self.accessor(self.index(primitive["indices"],"accessors"),False)
            if index_kind!="SCALAR" or index_origin.component_type not in (5121,5123,5125) or index_origin.normalized:
                self.fail("Indices must be unsigned integer SCALAR")
            faces=indices.reshape(-1).astype(np.int64)
            maximum={5121:255,5123:65535,5125:4294967295}[index_origin.component_type]
            if np.any(faces>=len(vertices)) or np.any(indices==maximum): self.fail("Triangle index is out of range")
        else: faces=np.arange(len(vertices),dtype=np.int64)
        if len(faces)%3: self.fail("TRIANGLES count must be divisible by three")
        mesh=MeshData(vertices,faces.reshape(-1,3),source_mesh_index=mesh_index,source_primitive_index=primitive_index)
        mesh.attribute_sources["POSITION"]=origin
        for name,value in attributes.items():
            if name=="POSITION": continue
            index=self.index(value,"accessors")
            data,attribute_source,shape=self.accessor(index,True)
            if len(data)!=len(vertices): self.fail(f"Attribute count mismatch: {name}")
            mesh.attribute_sources[name]=attribute_source
            if name in ("NORMAL","TANGENT"):
                if shape!=("VEC3" if name=="NORMAL" else "VEC4") or attribute_source.component_type!=5126:
                    self.fail(f"Invalid {name} attribute")
                if name=="NORMAL": mesh.vertex_normals=self.floating(data,attribute_source)
                else: mesh.tangents=self.floating(data,attribute_source)
            elif match:=re.fullmatch(r"(TEXCOORD|COLOR)_(\d+)",name):
                is_uv=match[1]=="TEXCOORD"
                if shape not in (("VEC2",) if is_uv else ("VEC3","VEC4")): self.fail(f"Invalid {name} shape")
                if attribute_source.component_type!=5126 and (attribute_source.component_type not in (5121,5123) or not attribute_source.normalized):
                    self.fail(f"Invalid {name} component type")
                target=mesh.uv_sets if is_uv else mesh.color_sets
                target[int(match[2])]=self.floating(data,attribute_source)
            elif name.startswith("_"): mesh.custom_attributes[name]=data
            else: self.fail(f"Unsupported vertex attribute {name}","unsupported_asset")
        for mapping in (mesh.uv_sets,mesh.color_sets):
            if mapping and sorted(mapping)!=list(range(max(mapping)+1)): self.fail("Attribute set indices must not have gaps")
        if "material" in primitive:
            mesh.material_index=self.index(primitive["material"],"materials")
        return mesh

    def number(self, value: JsonValue, label: str, minimum: float | None = None,
               maximum: float | None = None) -> float:
        if not isinstance(value,(int,float)) or isinstance(value,bool): self.fail(f"Invalid {label}")
        try:
            result=float(value)
        except OverflowError:
            self.fail(f"Non-finite {label}")
        if not math.isfinite(result): self.fail(f"Non-finite {label}")
        if (minimum is not None and result<minimum) or (maximum is not None and result>maximum): self.fail(f"Out-of-range {label}")
        return result

    def binding(self, value: JsonValue) -> TextureBinding:
        info=self.obj(value,"texture slot")
        index=self.index(info.get("index"),"textures")
        uv=self.integer(info.get("texCoord",0),"texture.texCoord")
        transform: TextureTransform | None = None
        extensions=self.obj(info.get("extensions",{}),"texture slot extensions")
        if "KHR_texture_transform" in extensions:
            item=self.obj(extensions["KHR_texture_transform"],"texture transform")
            offset=self.vector(item.get("offset",[0,0]),2,"texture offset")
            scale=self.vector(item.get("scale",[1,1]),2,"texture scale")
            override=self.integer(item["texCoord"],"texture transform texCoord") if "texCoord" in item else None
            transform=TextureTransform((float(offset[0]),float(offset[1])),
                                       self.number(item.get("rotation",0),"texture rotation"),
                                       (float(scale[0]),float(scale[1])),override)
        return TextureBinding(index,uv,transform)

    def resources(self, asset: Asset) -> None:
        for i,image in enumerate(self.tables["images"]):
            if "uri" in image and "bufferView" in image: self.fail("Image cannot have both URI and bufferView")
            if "uri" in image:
                payload,uri_mime=self.embedded(image["uri"])
                mime=image.get("mimeType",uri_mime)
                if mime!=uri_mime: self.fail("Image MIME conflicts with its data URI")
            else:
                index=self.index(image.get("bufferView"),"bufferViews")
                buffer,start,size,_=self.view(index);payload=buffer[start:start+size]
                mime=image.get("mimeType")
                if mime is None: self.fail("bufferView image requires mimeType")
            if mime not in ("image/png","image/jpeg"): self.fail("Only PNG/JPEG images are supported","unsupported_asset")
            if (mime=="image/png" and not payload.startswith(b"\x89PNG\r\n\x1a\n")) or (mime=="image/jpeg" and not payload.startswith(b"\xff\xd8\xff")):
                self.fail("Image MIME and encoded signature do not match")
            asset.images.append(ImageData(payload,cast("Literal['image/png', 'image/jpeg']",mime),i))
        if asset.images:
            self.source.notes.append("Encoded PNG/JPEG bytes preserved; pixel decoding and visual consistency are not verified.")
        for sampler in self.tables["samplers"]:
            mag=self.integer(sampler["magFilter"],"magFilter") if "magFilter" in sampler else None
            minimum=self.integer(sampler["minFilter"],"minFilter") if "minFilter" in sampler else None
            wrap_s=self.integer(sampler.get("wrapS",10497),"wrapS")
            wrap_t=self.integer(sampler.get("wrapT",10497),"wrapT")
            if mag not in (None,9728,9729) or minimum not in (None,9728,9729,9984,9985,9986,9987): self.fail("Invalid sampler filter")
            if wrap_s not in (33071,33648,10497) or wrap_t not in (33071,33648,10497): self.fail("Invalid sampler wrap")
            asset.samplers.append(SamplerData(mag,minimum,wrap_s,wrap_t))
        for i,texture in enumerate(self.tables["textures"]):
            image_index=self.index(texture.get("source"),"images")
            sampler_index=self.index(texture["sampler"],"samplers") if "sampler" in texture else None
            asset.textures.append(TextureData(image_index,sampler_index,i))
        for i,item in enumerate(self.tables["materials"]):
            pbr=self.obj(item.get("pbrMetallicRoughness",{}),"PBR material")
            color=self.vector(pbr.get("baseColorFactor",[1,1,1,1]),4,"baseColorFactor")
            emissive=self.vector(item.get("emissiveFactor",[0,0,0]),3,"emissiveFactor")
            if np.any(color<0) or np.any(color>1) or np.any(emissive<0) or np.any(emissive>1): self.fail("Color factor is out of range")
            alpha=item.get("alphaMode","OPAQUE")
            if alpha not in ("OPAQUE","MASK","BLEND"): self.fail("Invalid alphaMode")
            double_sided=item.get("doubleSided",False)
            if not isinstance(double_sided,bool): self.fail("doubleSided must be boolean")
            extensions=self.obj(item.get("extensions",{}),"material extensions")
            material=MaterialData(
                base_color_factor=cast(tuple[float,float,float,float],tuple(float(v) for v in color)),
                metallic_factor=self.number(pbr.get("metallicFactor",1),"metallicFactor",0,1),
                roughness_factor=self.number(pbr.get("roughnessFactor",1),"roughnessFactor",0,1),
                emissive_factor=cast(tuple[float,float,float],tuple(float(v) for v in emissive)),
                alpha_mode=cast("Literal['OPAQUE', 'MASK', 'BLEND']",alpha),
                alpha_cutoff=self.number(item.get("alphaCutoff",.5),"alphaCutoff",0),
                double_sided=double_sided,unlit="KHR_materials_unlit" in extensions,source_material_index=i)
            for name,slot,container in [("baseColorTexture","base_color",pbr),("metallicRoughnessTexture","metallic_roughness",pbr),
                                        ("normalTexture","normal",item),("occlusionTexture","occlusion",item),("emissiveTexture","emissive",item)]:
                if name in container: material.textures[cast(TextureSlot,slot)]=self.binding(container[name])
            if "normalTexture" in item:
                material.normal_scale=self.number(self.obj(item["normalTexture"],"normalTexture").get("scale",1),"normal scale")
            if "occlusionTexture" in item:
                material.occlusion_strength=self.number(self.obj(item["occlusionTexture"],"occlusionTexture").get("strength",1),"occlusion strength",0,1)
            asset.materials.append(material)
        for mesh in asset.meshes:
            if mesh.material_index is not None:
                for binding in asset.materials[mesh.material_index].textures.values():
                    uv=binding.texcoord
                    if binding.transform is not None and binding.transform.texcoord_override is not None:
                        uv=binding.transform.texcoord_override
                    if uv not in mesh.uv_sets: self.fail(f"Material references missing TEXCOORD_{uv}")

    def vector(self, value: JsonValue, width: int, label: str) -> FloatArray:
        items=self.arr(value,label)
        if len(items)!=width: self.fail(f"Invalid {label}")
        return np.asarray([self.number(v,label) for v in items],dtype=np.float64)

    def transform(self, node: Object) -> FloatArray:
        if "matrix" in node:
            if node.keys()&{"translation","rotation","scale"}: self.fail("Node cannot have both matrix and TRS")
            matrix=self.vector(node["matrix"],16,"node.matrix").reshape(4,4,order="F").copy()
            if not np.array_equal(matrix[3],[0,0,0,1]): self.fail("Node matrix must be affine")
            return matrix
        t=self.vector(node.get("translation",[0,0,0]),3,"translation")
        q=self.vector(node.get("rotation",[0,0,0,1]),4,"rotation")
        s=self.vector(node.get("scale",[1,1,1]),3,"scale")
        if not np.isclose(q@q,1,rtol=0,atol=1e-6): self.fail("Node rotation must be a unit quaternion")
        x,y,z,w=q
        rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                           [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                           [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],dtype=np.float64)
        matrix=np.eye(4,dtype=np.float64);matrix[:3,:3]=rotation*s;matrix[:3,3]=t
        return matrix

    def read(self) -> Asset:
        scenes=self.tables["scenes"]
        if "scenes" not in self.doc: self.fail("GLB has no scenes","unsupported_asset")
        if not scenes: self.fail("scenes cannot be empty")
        if "scene" in self.doc: scene=self.index(self.doc["scene"],"scenes")
        elif len(scenes)==1: scene=0
        else: self.fail("Multiple scenes require a default scene","unsupported_asset")
        self.source.selected_scene_index=scene
        node_table=self.tables["nodes"]
        parents: dict[int,int] = {}
        children: dict[int,list[int]] = {}
        for i,node in enumerate(node_table):
            children[i]=[self.index(v,"nodes") for v in self.arr(node.get("children",[]),"children")]
            if "mesh" in node: self.index(node["mesh"],"meshes")
            if "camera" in node: self.index(node["camera"],"cameras")
            for child in children[i]:
                if child in parents: self.fail("Node has multiple parents")
                parents[child]=i
        states: dict[int,int] = {}
        for start in range(len(node_table)):
            stack = [(start,False)]
            while stack:
                index,exiting = stack.pop()
                if exiting:
                    states[index]=2
                    continue
                if states.get(index)==1: self.fail("Node hierarchy contains a cycle")
                if states.get(index)==2: continue
                states[index]=1
                stack.append((index,True))
                stack.extend((child,False) for child in reversed(children[index]))
        roots=[self.index(v,"nodes") for v in self.arr(scenes[scene].get("nodes",[]),"scene.nodes")]
        selected: list[int] = []
        for root in roots:
            if root in parents: self.fail("Scene root has a parent")
        pending=list(reversed(roots))
        selected_set: set[int] = set()
        while pending:
            index=pending.pop()
            if index in selected_set: self.fail("Scene repeats a node")
            selected_set.add(index)
            selected.append(index)
            pending.extend(reversed(children[index]))
        mapping={original:i for i,original in enumerate(selected)}
        meshes: list[MeshData] = [];nodes: list[NodeData] = [];mesh_cache: dict[int,list[int]] = {}
        for original in selected:
            node=node_table[original];references=[]
            if "mesh" in node:
                mesh_index=self.index(node["mesh"],"meshes")
                if mesh_index not in mesh_cache:
                    primitives=self.arr(self.tables["meshes"][mesh_index].get("primitives"),"mesh.primitives")
                    if not primitives: self.fail("Mesh must contain a primitive")
                    mesh_cache[mesh_index]=[]
                    for p,primitive in enumerate(primitives):
                        mesh_cache[mesh_index].append(len(meshes))
                        meshes.append(self.geometry(mesh_index,p,self.obj(primitive,"primitive")))
                references=list(mesh_cache[mesh_index])
            parent=parents.get(original)
            nodes.append(NodeData(mapping[parent] if parent is not None else None,references,self.transform(node),original))
        if not meshes or sum(len(m.faces) for m in meshes)==0: self.fail("Selected scene has no triangles","unsupported_asset")
        self.source.metadata={"asset":self.doc.get("asset"),"nodes":[node_table[i] for i in selected],"cameras":self.doc.get("cameras",[])}
        self.source.metadata["scene"]={k:v for k,v in scenes[scene].items() if k in ("name","extras")}
        self.source.metadata["accessors"]=[{k:v for k,v in item.items() if k in ("name","extras")} for item in self.tables["accessors"]]
        self.source.metadata["primitives"]=[
            [{k:v for k,v in self.obj(p,"primitive").items() if k in ("attributes","indices","name","extras")}
             for p in self.arr(mesh.get("primitives"),"mesh.primitives")]
            for mesh in self.tables["meshes"]]
        for name in ("meshes","materials","images","textures","samplers"):
            self.source.metadata[name]=[{k:v for k,v in item.items() if k in ("name","extras")} for item in self.tables[name]]
        asset=Asset(self.source,meshes,nodes)
        self.resources(asset)
        return asset


def read_glb(raw: bytes, source: AssetSource) -> Asset:
    """Read the explicitly supported static, self-contained GLB subset."""
    return _Reader(raw,source).read()
