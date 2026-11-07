"""Freeze a few local representative assets using source-only expectations.

This preparation utility never imports the product reader. Existing baselines
are not overwritten; use a new output directory when proposing a new baseline.
"""

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import struct

PROJECT = Path(__file__).resolve().parents[1]
ALLOWED_EXTENSIONS = {"KHR_texture_transform", "KHR_materials_unlit"}


def glb_json(path):
    with path.open('rb') as stream:
        magic,version,total,size,kind=struct.unpack('<4sIIII',stream.read(20))
        if magic!=b'glTF' or version!=2 or kind!=0x4e4f534a:
            raise ValueError(f'Not an ordinary GLB2 source: {path}')
        return json.loads(stream.read(size))


def selected_geometry(doc):
    scenes=doc.get('scenes',[])
    scene=doc.get('scene',0 if len(scenes)==1 else None)
    if scene is None or not scenes: raise ValueError('Source needs a default/unique scene')
    visited=[]
    def walk(index):
        if index in visited: raise ValueError('Source hierarchy is not a tree')
        visited.append(index)
        for child in doc['nodes'][index].get('children',[]): walk(child)
    for index in scenes[scene].get('nodes',[]): walk(index)
    source_meshes=sorted({doc['nodes'][i]['mesh'] for i in visited if 'mesh' in doc['nodes'][i]})
    primitives=[p for index in source_meshes for p in doc['meshes'][index]['primitives']]
    return visited,primitives


def glb_expected(path,doc):
    nodes,primitives=selected_geometry(doc)
    vertices=sum(doc['accessors'][p['attributes']['POSITION']]['count'] for p in primitives)
    faces=sum(doc['accessors'][p['indices'] if 'indices' in p else p['attributes']['POSITION']]['count']//3 for p in primitives)
    raw=path.read_bytes();offset=12;blob=b''
    while offset<len(raw):
        size,kind=struct.unpack_from('<II',raw,offset);offset+=8
        if kind==0x004e4942: blob=raw[offset:offset+size]
        offset+=size
    def payload(index):
        buffer=doc['buffers'][index]
        if 'uri' not in buffer:return blob
        return base64.b64decode(buffer['uri'].split(',',1)[1])
    primitive=primitives[0]
    if 'indices' in primitive:
        accessor=doc['accessors'][primitive['indices']];view=doc['bufferViews'][accessor['bufferView']]
        start=view.get('byteOffset',0)+accessor.get('byteOffset',0)
        indices=struct.unpack_from('<3'+{5121:'B',5123:'H',5125:'I'}[accessor['componentType']],payload(view['buffer']),start)
    else: indices=(0,1,2)
    accessor=doc['accessors'][primitive['attributes']['POSITION']];view=doc['bufferViews'][accessor['bufferView']]
    start=view.get('byteOffset',0)+accessor.get('byteOffset',0);stride=view.get('byteStride',12)
    if accessor['componentType']!=5126: raise ValueError('Representative source requires FLOAT positions')
    triangle=[list(struct.unpack_from('<3f',payload(view['buffer']),start+i*stride)) for i in indices]
    image_hashes=[]
    for image in doc.get('images',[]):
        if 'uri' in image:
            encoded=base64.b64decode(image['uri'].split(',',1)[1])
        else:
            view=doc['bufferViews'][image['bufferView']];start=view.get('byteOffset',0)
            encoded=payload(view['buffer'])[start:start+view['byteLength']]
        image_hashes.append(hashlib.sha256(encoded).hexdigest())
    return {'format':'glb','mesh_count':len(primitives),'vertex_count':vertices,'face_count':faces,
            'instance_count':sum('mesh' in doc['nodes'][i] for i in nodes),'first_triangle':triangle,
            'primitive_materials':[p.get('material') for p in primitives],'image_sha256':image_hashes,
            'selected_scene_index':doc.get('scene',0),'material_count':len(doc.get('materials',[]))}


def stl_expected(path):
    raw=path.read_bytes()
    if len(raw)>=84 and 84+50*struct.unpack_from('<I',raw,80)[0]==len(raw):
        count=struct.unpack_from('<I',raw,80)[0]
        if not count or b'COLOR=' in raw[:80]: raise ValueError('Source outside ordinary STL scope')
        values=struct.unpack_from('<12fH',raw,84)
        if values[12]: raise ValueError('Source has nonzero attributes')
        triangle=[list(values[3:6]),list(values[6:9]),list(values[9:12])];normal=list(values[:3]);encoding='binary'
    else:
        lines=[line.split() for line in raw.decode('ascii').splitlines()]
        facets=[tokens[2:] for tokens in lines if tokens[:2]==['facet','normal']]
        points=[tokens[1:] for tokens in lines if tokens[:1]==['vertex']]
        if not facets or len(points)!=3*len(facets): raise ValueError('Source is not a complete triangle listing')
        count=len(facets);triangle=[[float(v) for v in point] for point in points[:3]]
        normal=[float(v) for v in facets[0]];encoding='ascii'
    if not all(math.isfinite(v) for row in triangle+[normal] for v in row): raise ValueError('Non-finite source')
    return {'format':'stl','encoding':encoding,'mesh_count':1,'face_count':count,'instance_count':1,
            'first_triangle':triangle,'first_normal':normal}


def write_new(path,value,jsonl=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        if jsonl:
            for row in value: stream.write(json.dumps(row,sort_keys=True)+'\n')
        else: stream.write(json.dumps(value,indent=2,sort_keys=True)+'\n')


def prepare(objaverse,thingi,output,thingi_provenance):
    source=json.loads((objaverse/'manifest.json').read_text())
    thingi_metadata=json.loads(thingi_provenance.read_text())
    candidates=[]
    for record in source['assets']:
        path=objaverse/record['path'];doc=glb_json(path)
        if set(doc.get('extensionsUsed',[]))-ALLOWED_EXTENSIONS or doc.get('animations'): continue
        if any('skin' in node for node in doc.get('nodes',[])): continue
        if any(p.get('targets') or p.get('mode',4)!=4 for m in doc.get('meshes',[]) for p in m['primitives']):continue
        if any('sparse' in a for a in doc.get('accessors',[])):continue
        nodes,primitives=selected_geometry(doc)
        if not primitives: continue
        features={record['category']}
        if len(nodes)>1: features.add('hierarchy')
        if len(primitives)>1: features.add('multiple-primitives')
        if doc.get('images'): features.add('images')
        if any('TEXCOORD_1' in p['attributes'] for p in primitives):features.add('multiple-uv')
        candidates.append((record,doc,features))
    chosen=[];covered=set()
    while candidates and len(chosen)<4:
        candidates.sort(key=lambda item:(-len(item[2]-covered),item[0]['size_bytes'],item[0]['uid']))
        record,doc,features=candidates.pop(0);covered.update(features)
        path=objaverse/record['path'];expected=glb_expected(path,doc)
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=record['sha256']:raise ValueError(f'Source hash mismatch: {path}')
        ref={'source_id':'objaverse-1.0','asset_id':record['uid'],'relative_path':record['path'],'expected_sha256':actual}
        expected.update(source_id=ref['source_id'],asset_id=ref['asset_id'],status='ok',features=sorted(features),
                        provenance={'revision':source['revision'],'url':record.get('url'),'license':record.get('metadata',{}).get('license')})
        chosen.append((ref,expected))
    with os.scandir(thingi/'raw_meshes') as entries:
        names=sorted(e.name for e in entries if e.name.lower().endswith('.stl') and not e.name.startswith('._'))
    encodings=set()
    for name in names:
        path=thingi/'raw_meshes'/name
        if path.stem not in thingi_metadata['assets']:continue
        if path.stat().st_size>2*1024*1024:continue
        try:expected=stl_expected(path)
        except (ValueError,UnicodeError,struct.error):continue
        if expected['encoding'] in encodings:continue
        encodings.add(expected['encoding'])
        ref={'source_id':'thingi10k','asset_id':path.stem,'relative_path':'raw_meshes/'+name,
             'expected_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        provenance={k:v for k,v in thingi_metadata.items() if k!='assets'}
        provenance.update(thingi_metadata['assets'][path.stem])
        expected.update(source_id=ref['source_id'],asset_id=ref['asset_id'],status='ok',provenance=provenance)
        chosen.append((ref,expected))
        if len(encodings)==2:break
    if len(chosen)<6:raise ValueError(f'Could not select 4 GLBs and 2 STL encodings; selected {len(chosen)}')
    small=[chosen[0],*chosen[-2:]]
    for label,items in [('real-small',small),('registered',chosen)]:
        write_new(output/'manifests'/f'{label}.jsonl',[r for r,_ in items],True)
        write_new(output/'expectations'/f'{label}.json',{'assets':[e for _,e in items],
                  'selection':'4 diverse GLBs and ASCII/binary STL; small subset uses one GLB and both STL encodings.'})
    storage=PROJECT/'configs/storage.local.json'
    if not storage.exists():write_new(storage,{'objaverse-1.0':str(objaverse),'thingi10k':str(thingi)})
    print(json.dumps({'small':len(small),'registered':len(chosen),'assets':[e for _,e in chosen]},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objaverse-root',required=True,type=Path)
    parser.add_argument('--thingi-root',required=True,type=Path)
    parser.add_argument('--output-root',type=Path,default=PROJECT/'tests')
    parser.add_argument('--thingi-provenance',type=Path,default=PROJECT/'tests/expectations/thingi-provenance.json')
    args=parser.parse_args()
    prepare(args.objaverse_root.resolve(),args.thingi_root.resolve(),args.output_root.resolve(),args.thingi_provenance.resolve())
