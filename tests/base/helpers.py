"""Small independent byte builders; never call the product importer."""

from pathlib import Path
import tempfile
import struct
import json
import math
import base64

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNwaDgAAAKEAYEml6crAAAAAElFTkSuQmCC")
JPEG = base64.b64decode("/9j/4AAQSkZJRgABAQAASABIAAD/4QBMRXhpZgAATU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAAaADAAQAAAABAAAAAQAAAAD/7QA4UGhvdG9zaG9wIDMuMAA4QklNBAQAAAAAAAA4QklNBCUAAAAAABDUHYzZjwCyBOmACZjs+EJ+/8AAEQgAAQABAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/bAEMAAgICAgICAwICAwUDAwMFBgUFBQUGCAYGBgYGCAoICAgICAgKCgoKCgoKCgwMDAwMDA4ODg4ODw8PDw8PDw8PD//bAEMBAgICBAQEBwQEBxALCQsQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEP/dAAQAAf/aAAwDAQACEQMRAD8A4+iiiv6sP5rP/9k=")

from meshengineer.core.assets import AssetRef
from meshengineer.io.loader import load_asset


def ascii_stl(triangles, normals=None):
    lines = ["solid sample"]
    for i, triangle in enumerate(triangles):
        normal = normals[i] if normals is not None else (0, 0, 1)
        lines.append("facet normal " + " ".join(str(v) for v in normal))
        lines.append("outer loop")
        lines.extend("vertex " + " ".join(str(v) for v in p) for p in triangle)
        lines.extend(["endloop", "endfacet"])
    return ("\n".join(lines + ["endsolid sample"]) + "\n").encode("ascii")


def load_bytes(raw: bytes, suffix: str = ".stl"):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        p = root / ("asset" + suffix)
        p.write_bytes(raw)
        try:
            asset = load_asset(AssetRef("test", "asset", p.name), {"test": root})
        finally:
            if p.read_bytes() != raw:
                raise AssertionError("Importer modified source bytes")
        return asset


def binary_stl(triangles, normals=None, header=b"manual", attribute=0):
    records=[]
    for i,triangle in enumerate(triangles):
        normal=normals[i] if normals is not None else (0,0,1)
        values=[*normal,*(v for point in triangle for v in point)]
        records.append(struct.pack("<12fH",*values,attribute))
    return header.ljust(80,b"\0")+struct.pack("<I",len(triangles))+b"".join(records)


def pack_glb(document, blob=b"", extra_chunks=()):
    text=json.dumps(document,separators=(",",":"),allow_nan=False).encode()
    text+=b" "*((-len(text))%4)
    body=struct.pack("<II",len(text),0x4e4f534a)+text
    if blob:
        padded=blob+b"\0"*((-len(blob))%4)
        body+=struct.pack("<II",len(padded),0x004e4942)+padded
    for kind,data in extra_chunks:
        body+=struct.pack("<II",len(data),kind)+data
    return struct.pack("<4sII",b"glTF",2,12+len(body))+body


def simple_glb(index_type=5123):
    positions=struct.pack("<12f",0,0,0,1,0,0,1,1,0,0,1,0)
    code={5121:"B",5123:"H",5125:"I"}[index_type]
    indices=struct.pack("<6"+code,0,1,2,0,2,3)
    blob=positions+indices
    doc={"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
         "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],
         "buffers":[{"byteLength":len(blob)}],
         "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":48},{"buffer":0,"byteOffset":48,"byteLength":len(indices)}],
         "accessors":[{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},
                      {"bufferView":1,"componentType":index_type,"count":6,"type":"SCALAR"}]}
    return doc,blob


def rich_glb():
    doc={"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],
         "nodes":[{"children":[1,2],"translation":[10,0,0]},
                  {"mesh":0,"translation":[2,3,4],"rotation":[0,0,math.sqrt(.5),math.sqrt(.5)],"scale":[2,1,1]},
                  {"mesh":0,"matrix":[1,0,0,0,0,2,0,0,0,0,3,0,-1,0,2,1]}],
         "meshes":[{"primitives":[]}],"buffers":[{}],"bufferViews":[],"accessors":[]}
    blob=bytearray()
    def attribute(values,shape,fmt="f",component=5126,normalized=False):
        width={"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}[shape]
        size=struct.calcsize("<"+fmt*width);stride=(size+3)//4*4
        start=len(blob)
        for row in values:
            blob.extend(struct.pack("<"+fmt*width,*row));blob.extend(b"\0"*(stride-size))
        view={"buffer":0,"byteOffset":start,"byteLength":len(blob)-start,"byteStride":stride}
        doc["bufferViews"].append(view)
        item={"bufferView":len(doc["bufferViews"])-1,"componentType":component,"type":shape,"count":len(values)}
        if normalized:item["normalized"]=True
        if shape=="VEC3" and values:
            item["min"]=[min(v[i] for v in values) for i in range(3)]
            item["max"]=[max(v[i] for v in values) for i in range(3)]
        doc["accessors"].append(item)
        return len(doc["accessors"])-1
    for z in [0,1]:
        attrs={"POSITION":attribute([(0,0,z),(1,0,z),(0,2,z)],"VEC3"),
               "NORMAL":attribute([(0,0,1)]*3,"VEC3"),
               "TANGENT":attribute([(1,0,0,1),(1,0,0,-1),(1,0,0,1)],"VEC4"),
               "TEXCOORD_0":attribute([(0,0),(1,0),(0,1)],"VEC2"),
               "TEXCOORD_1":attribute([(.25,.5),(.5,.5),(.25,.75)],"VEC2"),
               "COLOR_0":attribute([(1,.25,.5),(.5,1,.25),(.25,.5,1)] if z==0 else [(.25,.5,.75,1)]*3,"VEC3" if z==0 else "VEC4"),
               "_LABEL":attribute([(11,),(12,),(13,)],"SCALAR","H",5123),
               "_VALUE":attribute([(.25,.5),(.5,.75),(.75,1)],"VEC2")}
        doc["meshes"][0]["primitives"].append({"attributes":attrs})
    doc["buffers"][0]["byteLength"]=len(blob)
    return doc,bytes(blob)


def material_glb():
    doc,blob=rich_glb();offset=len(blob);blob+=PNG
    doc['bufferViews'].append({'buffer':0,'byteOffset':offset,'byteLength':len(PNG)})
    doc['buffers'][0]['byteLength']=len(blob)
    doc['images']=[{'bufferView':len(doc['bufferViews'])-1,'mimeType':'image/png'},
                   {'uri':'data:image/jpeg;base64,'+base64.b64encode(JPEG).decode()}]
    doc['textures']=[{'source':0,'sampler':0},{'source':1}]
    doc['samplers']=[{'magFilter':9729,'minFilter':9987,'wrapS':33071,'wrapT':33648},{}]
    doc['extensionsUsed']=['KHR_texture_transform','KHR_materials_unlit']
    doc['materials']=[{'pbrMetallicRoughness':{'baseColorFactor':[.25,.5,.75,1],'metallicFactor':.3,'roughnessFactor':.7,
                      'baseColorTexture':{'index':0,'texCoord':0,'extensions':{'KHR_texture_transform':{'offset':[.1,.2],'rotation':.5,'scale':[2,3],'texCoord':1}}},
                      'metallicRoughnessTexture':{'index':1,'texCoord':1}},
                      'normalTexture':{'index':1,'scale':.5},'occlusionTexture':{'index':0,'texCoord':1,'strength':.75},
                      'emissiveTexture':{'index':1},'emissiveFactor':[.1,.2,.3],'alphaMode':'MASK','alphaCutoff':.25,'doubleSided':True},
                     {'pbrMetallicRoughness':{'baseColorTexture':{'index':1}},'extensions':{'KHR_materials_unlit':{}}}]
    for i,primitive in enumerate(doc['meshes'][0]['primitives']): primitive['material']=i
    return doc,blob


def compression_glb(name, required=False, missing=False, broken=False):
    data=json.loads((Path(__file__).resolve().parents[1]/'fixtures/compression.json').read_text())
    if name=='KHR_draco_mesh_compression':
        item=data['draco'];doc=item['plain_document'];blob=base64.b64decode(item['plain_binary_base64'])
        compressed=base64.b64decode(item['binary_base64']);start=len(blob);blob+=compressed
        doc['buffers'][0]={'byteLength':len(blob)}
        doc['bufferViews'].append({'buffer':0,'byteOffset':start,'byteLength':118})
        doc['meshes'][0]['primitives'][0]['extensions']={name:{'bufferView':2,'attributes':{'NORMAL':0,'POSITION':1}}}
        if missing:
            for accessor in doc['accessors']: accessor.pop('bufferView',None)
        if broken: doc['bufferViews'][1]['byteLength']=1
    else:
        doc,blob=simple_glb();compressed=base64.b64decode(data['meshopt']['base64']);start=len(blob);blob+=compressed
        doc['buffers'][0]['byteLength']=len(blob)
        doc['bufferViews'][0]['extensions']={name:{'buffer':0,'byteOffset':start,'byteLength':len(compressed),'byteStride':12,'count':4,'mode':'ATTRIBUTES','filter':'NONE'}}
        if missing:
            doc['buffers'].append({'byteLength':48,'extensions':{name:{'fallback':True}}})
            doc['bufferViews'][0]['buffer']=1
        if broken: doc['bufferViews'][0]['byteLength']=47
    doc['extensionsUsed']=[name]
    if required: doc['extensionsRequired']=[name]
    return doc,blob
