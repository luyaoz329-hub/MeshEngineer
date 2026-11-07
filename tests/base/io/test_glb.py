"""GLB behavior at load_asset, with independently constructed bytes."""

import unittest
import copy
import struct
import base64
import json
from pathlib import Path
import numpy as np

from meshengineer.core.imports import AssetImportError
from tests.base.helpers import load_bytes, pack_glb, simple_glb, rich_glb, material_glb, compression_glb, PNG, JPEG


class GlbTests(unittest.TestCase):
    def test_u02_legal_legacy_glb1_container_is_explicitly_unsupported(self):
        # Independent GLB1 layout: five uint32 header fields, padded JSON, no payload.
        # Khronos extensions/1.0/Khronos/KHR_binary_glTF and glTF1 asset schema.
        document={'asset':{'version':'1.0'},'extensionsUsed':['KHR_binary_glTF']}
        text=json.dumps(document,separators=(',',':')).encode('utf-8');text+=b' '*((-len(text))%4)
        raw=struct.pack('<4s4I',b'glTF',1,20+len(text),len(text),0)+text
        with self.assertRaises(AssetImportError) as result:load_bytes(raw,'.glb')
        self.assertEqual(result.exception.code,'unsupported_asset');self.assertIn('version 1',result.exception.message)

    def test_u05_real_basisu_ktx2_is_rejected_with_or_without_png_fallback(self):
        encoded=json.loads((Path(__file__).resolve().parents[2]/'fixtures/unsupported-encodings.json').read_text())['basisu']['base64']
        raw=base64.b64decode(encoded)
        self.assertEqual(raw[:12],b'\xabKTX 20\xbb\r\n\x1a\n')
        self.assertEqual(struct.unpack_from('<I',raw,44)[0],1)  # BasisLZ supercompression.
        for required in [False,True]:
            doc,blob=material_glb();doc['extensionsUsed'].append('KHR_texture_basisu')
            doc['images'].append({'uri':'data:image/ktx2;base64,'+encoded})
            doc['textures'][0]['extensions']={'KHR_texture_basisu':{'source':2}}
            if required:
                doc['extensionsRequired']=['KHR_texture_basisu'];doc['textures'][0].pop('source')
            self.assert_code(doc,blob,'unsupported_asset')

    def test_g05_normalized_integer_uvs_keep_exact_set_and_scale(self):
        for component,fmt,middle,maximum in [(5121,'B',128,255),(5123,'H',32768,65535)]:
            doc,blob=simple_glb();offset=len(blob)
            rows=[(0,maximum),(middle,0),(maximum,middle),(middle,maximum)]
            element=struct.calcsize('<2'+fmt)
            encoded=b''.join(struct.pack('<2'+fmt,*row)+b'\0'*(4-element) for row in rows)
            blob+=encoded;doc['buffers'][0]['byteLength']=len(blob)
            doc['bufferViews'].append({'buffer':0,'byteOffset':offset,'byteLength':len(encoded),'byteStride':4})
            doc['accessors'].append({'bufferView':2,'count':4,'type':'VEC2','componentType':component,'normalized':True})
            doc['meshes'][0]['primitives'][0]['attributes']['TEXCOORD_0']=2
            mesh=load_bytes(pack_glb(doc,blob),'.glb').meshes[0]
            np.testing.assert_allclose(mesh.uv_sets[0],[[0,1],[middle/maximum,0],[1,middle/maximum],[middle/maximum,1]],rtol=0,atol=1e-15)

    def test_u08_relative_alignment_and_declared_stride_bounds(self):
        doc,blob=simple_glb();blob=blob[:48]+b'xx'+blob[48:]
        doc['buffers'][0]['byteLength']=len(blob)
        doc['bufferViews'][1].update(byteOffset=49,byteLength=13)
        doc['accessors'][1]['byteOffset']=1
        self.assert_code(doc,blob,'invalid_asset')
        doc,original=simple_glb()
        blob=b''.join(original[i:i+12]+b'\0'*244 for i in range(0,48,12))+original[48:]
        doc['buffers'][0]['byteLength']=len(blob)
        doc['bufferViews'][0].update(byteLength=1024,byteStride=256)
        doc['bufferViews'][1]['byteOffset']=1024
        self.assert_code(doc,blob,'invalid_asset')

    def test_g17_scene_primitive_and_accessor_names_extras_survive(self):
        doc,blob=simple_glb()
        for item,name in [(doc['scenes'][0],'scene'),(doc['meshes'][0]['primitives'][0],'primitive'),(doc['accessors'][0],'position')]:
            item.update(name=name,extras={'source':name})
        asset=load_bytes(pack_glb(doc,blob),'.glb')
        self.assertEqual(asset.source.metadata['scene'],{'name':'scene','extras':{'source':'scene'}})
        self.assertEqual(asset.source.metadata['primitives'][0],[{'name':'primitive','extras':{'source':'primitive'},'attributes':{'POSITION':0},'indices':1}])
        self.assertEqual(asset.source.metadata['accessors'][0],{'name':'position','extras':{'source':'position'}})

    def test_g12_deep_scene_hierarchy_is_iterative(self):
        doc,blob=simple_glb();depth=1200
        doc['nodes']=[{'children':[i+1]} for i in range(depth)]+[{'mesh':0}]
        asset=load_bytes(pack_glb(doc,blob),'.glb')
        self.assertEqual(len(asset.nodes),depth+1)
        self.assertEqual(asset.nodes[-1].parent_index,depth-1)
        self.assertEqual(asset.nodes[-1].mesh_indices,[0])
        doc['nodes'][-1]['children']=[0]
        self.assert_code(doc,blob,'invalid_asset')

    def test_g14_both_reflection_and_zero_scale_remain_local(self):
        for scale,expected in [([-1,1,1],[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),
                               ([0,1,1],[[0,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])]:
            doc,blob=rich_glb();doc['nodes'][1]={'mesh':0,'scale':scale}
            asset=load_bytes(pack_glb(doc,blob),'.glb')
            np.testing.assert_array_equal(asset.nodes[1].local_transform,expected)
            np.testing.assert_array_equal(asset.meshes[0].faces,[[0,1,2]])
            np.testing.assert_array_equal(asset.meshes[0].vertex_normals,[[0,0,1]]*3)

    def test_g16_seams_nonmanifold_intersections_and_normals_are_not_cleaned(self):
        variants=[([(0,0,0),(1,0,0),(0,1,0),(0,0,0)],[(0,1,2),(3,2,1)]),
                  ([(0,0,0),(1,0,0),(0,1,0),(0,-1,0),(0,0,1)],[(0,1,2),(1,0,3),(0,1,4)]),
                  ([(-1,0,0),(1,0,0),(0,2,0),(0,-1,-1),(0,1,1),(0,1,-1)],[(0,1,2),(3,4,5)]),
                  ([(0,0,0),(1,0,0),(0,1,0),(8,0,0),(9,0,0),(8,1,0)],[(0,1,2),(3,4,5)])]
        for points,faces in variants:
            with self.subTest(points=points):
                doc,_=simple_glb();positions=struct.pack('<'+'f'*(3*len(points)),*(v for p in points for v in p))
                indices=struct.pack('<'+'H'*(3*len(faces)),*(v for f in faces for v in f))
                # Distinct attributes at equal seam coordinates, including finite non-unit normals.
                normals=[(0,0,0),(0,0,2),(0,0,-1)]+[(0,0,1)]*(len(points)-3)
                blob=positions+indices;bpad=(-len(blob))%4;blob+=b'\0'*bpad;offset=len(blob)
                blob+=struct.pack('<'+'f'*(3*len(normals)),*(v for p in normals for v in p))
                doc['buffers'][0]['byteLength']=len(blob)
                doc['bufferViews']=[{'buffer':0,'byteLength':len(positions)},
                                    {'buffer':0,'byteOffset':len(positions),'byteLength':len(indices)},
                                    {'buffer':0,'byteOffset':offset,'byteLength':12*len(points)}]
                doc['accessors'][0].update(count=len(points),min=[min(p[i] for p in points) for i in range(3)],max=[max(p[i] for p in points) for i in range(3)])
                doc['accessors'][1]['count']=3*len(faces)
                doc['accessors'].append({'bufferView':2,'count':len(points),'componentType':5126,'type':'VEC3'})
                doc['meshes'][0]['primitives'][0]['attributes']['NORMAL']=2
                mesh=load_bytes(pack_glb(doc,blob),'.glb').meshes[0]
                np.testing.assert_array_equal(mesh.vertices,points);np.testing.assert_array_equal(mesh.faces,faces)
                np.testing.assert_array_equal(mesh.vertex_normals,normals)

    def test_g17_names_cameras_and_extras_are_retained(self):
        doc,blob=simple_glb();doc['cameras']=[{'type':'perspective','perspective':{'yfov':.5,'znear':.1}}]
        doc['nodes']=[{'mesh':0,'name':'same','children':[1]},{'camera':0,'name':'same','extras':{'license':'https://example.org/license'}}]
        doc['meshes'][0]['name']='same'
        asset=load_bytes(pack_glb(doc,blob),'.glb')
        self.assertEqual(len(asset.nodes),2)
        self.assertEqual(asset.source.metadata['cameras'],doc['cameras'])
        self.assertEqual(asset.source.metadata['nodes'],doc['nodes'])
        self.assertEqual(asset.source.metadata['meshes'],[{'name':'same'}])

    def test_u08_additional_independent_bounds_shapes_and_references(self):
        changes=[lambda d:d['meshes'][0]['primitives'][0]['attributes'].update(POSITION=len(d['accessors'])),
                 lambda d:d['nodes'][0].update(camera=0),
                 lambda d:d['accessors'][1].update(type='VEC2'),
                 lambda d:d['bufferViews'][0].update(byteLength=47)]
        for change in changes:
            doc,blob=simple_glb();change(doc);self.assert_code(doc,blob,'invalid_asset')
        for semantic,shape in [('NORMAL','VEC2'),('TANGENT','VEC3'),('TEXCOORD_0','VEC3'),('COLOR_0','VEC2')]:
            doc,blob=rich_glb();index=doc['meshes'][0]['primitives'][0]['attributes'][semantic]
            doc['accessors'][index]['type']=shape;self.assert_code(doc,blob,'invalid_asset')
        doc,blob=rich_glb();doc['accessors'][1]['count']=2;self.assert_code(doc,blob,'invalid_asset')
        doc,blob=simple_glb();blob=blob[:48]+struct.pack('<6H',4,1,2,0,2,3);self.assert_code(doc,blob,'invalid_asset')

    def test_u09_each_float_attribute_and_transform_rejects_nonfinite(self):
        for semantic in ['POSITION','NORMAL','TANGENT','TEXCOORD_0','COLOR_0','_VALUE']:
            for value in [float('nan'),float('inf'),-float('inf')]:
                with self.subTest(semantic=semantic,value=value):
                    doc,blob=rich_glb();index=doc['meshes'][0]['primitives'][0]['attributes'][semantic]
                    view=doc['bufferViews'][doc['accessors'][index]['bufferView']];offset=view['byteOffset']
                    blob=blob[:offset]+struct.pack('<f',value)+blob[offset+4:]
                    self.assert_code(doc,blob,'invalid_asset')
        # Overflow notation is legal JSON but cannot form a finite local transform.
        doc,blob=simple_glb();doc['nodes'][0]['translation']=[1e300,0,0]
        raw=pack_glb(doc,blob).replace(b'1e+300',b'1e+999')
        with self.assertRaises(AssetImportError) as result:load_bytes(raw,'.glb')
        self.assertEqual(result.exception.code,'invalid_asset')

    def test_u09_huge_json_numbers_are_asset_failures(self):
        for field,exponent in [('translation',400),('alphaCutoff',400),('metallicFactor',100)]:
            with self.subTest(field=field,exponent=exponent):
                doc,blob=material_glb()
                if field=='translation': doc['nodes'][0]['translation']=[10**exponent,0,0]
                elif field=='metallicFactor': doc['materials'][0]['pbrMetallicRoughness']['metallicFactor']=10**exponent
                else: doc['materials'][0]['alphaCutoff']=10**exponent
                self.assert_code(doc,blob,'invalid_asset')

    def test_u06_real_compression_streams_and_ordinary_fallback_matrix(self):
        for name in ['KHR_draco_mesh_compression','EXT_meshopt_compression','KHR_meshopt_compression']:
            with self.subTest(extension=name):
                doc,blob=compression_glb(name);asset=load_bytes(pack_glb(doc,blob),'.glb')
                self.assertEqual(len(asset.meshes[0].faces),12 if name.startswith('KHR_draco') else 2)
                self.assertTrue(any(name in note for note in asset.source.notes))
                for options,code in [({'required':True},'unsupported_asset'),({'missing':True},'unsupported_asset'),({'broken':True},'invalid_asset')]:
                    doc,blob=compression_glb(name,**options);self.assert_code(doc,blob,code)

    def test_u08_material_binding_and_image_faults(self):
        modifications=[lambda d:d['meshes'][0]['primitives'][0].update(material=9),
                       lambda d:d['textures'][0].update(source=9),lambda d:d['textures'][0].update(sampler=9),
                       lambda d:d['materials'][0]['pbrMetallicRoughness']['baseColorTexture'].update(index=9),
                       lambda d:d['materials'][0]['pbrMetallicRoughness']['baseColorTexture']['extensions']['KHR_texture_transform'].update(texCoord=9),
                       lambda d:d['images'][0].update(mimeType='image/jpeg'),
                       lambda d:d['bufferViews'][-1].update(byteLength=999999)]
        for change in modifications:
            doc,blob=material_glb();change(doc);self.assert_code(doc,blob,'invalid_asset')

    def test_b10_image_bytes_outlive_parser_and_source(self):
        doc,blob=material_glb();asset=load_bytes(pack_glb(doc,blob),'.glb')
        import gc
        del doc,blob;gc.collect()
        self.assertEqual(asset.images[0].data,PNG);self.assertEqual(asset.images[1].data,JPEG)

    def test_g09_g10_g11_material_images_and_extensions(self):
        doc,blob=material_glb();asset=load_bytes(pack_glb(doc,blob),'.glb')
        self.assertEqual([m.material_index for m in asset.meshes],[0,1])
        material=asset.materials[0]
        self.assertEqual(material.base_color_factor,(.25,.5,.75,1))
        self.assertEqual(material.metallic_factor,.3);self.assertEqual(material.roughness_factor,.7)
        self.assertEqual(material.alpha_mode,'MASK');self.assertTrue(material.double_sided)
        self.assertEqual(material.alpha_cutoff,.25);self.assertEqual(material.emissive_factor,(.1,.2,.3))
        self.assertEqual(material.normal_scale,.5);self.assertEqual(material.occlusion_strength,.75)
        self.assertEqual(set(material.textures),{'base_color','metallic_roughness','normal','occlusion','emissive'})
        self.assertEqual({slot:(binding.texture_index,binding.texcoord) for slot,binding in material.textures.items()},
                         {'base_color':(0,0),'metallic_roughness':(1,1),'normal':(1,0),'occlusion':(0,1),'emissive':(1,0)})
        transform=material.textures['base_color'].transform
        self.assertEqual(transform.offset,(.1,.2));self.assertEqual(transform.scale,(2,3));self.assertEqual(transform.rotation,.5)
        self.assertEqual(transform.texcoord_override,1);self.assertTrue(asset.materials[1].unlit)
        self.assertEqual(asset.images[0].data,PNG);self.assertEqual(asset.images[1].data,JPEG)
        self.assertIsNone(asset.samplers[1].min_filter);self.assertIsNone(asset.textures[1].sampler_index)
        self.assertTrue(any('not verified' in note for note in asset.source.notes))
        self.assertEqual(asset.meshes[0].uv_sets[0].tolist(),[[0,0],[1,0],[0,1]])

    def assert_code(self,doc,blob,code):
        with self.assertRaises(AssetImportError) as result: load_bytes(pack_glb(doc,blob),".glb")
        self.assertEqual(result.exception.code,code,result.exception.message)

    def test_g01_indexed_glb_preserves_geometry_and_types(self):
        doc,blob=simple_glb()
        asset=load_bytes(pack_glb(doc,blob),".GLB")
        mesh=asset.meshes[0]
        self.assertEqual(mesh.vertices.tolist(),[[0,0,0],[1,0,0],[1,1,0],[0,1,0]])
        self.assertEqual(mesh.faces.tolist(),[[0,1,2],[0,2,3]])
        self.assertEqual(mesh.vertices.dtype,np.dtype('float64'))
        self.assertEqual(mesh.faces.dtype,np.dtype('int64'))
        self.assertIsNone(mesh.vertex_normals)
        self.assertEqual(asset.source.selected_scene_index,0)

    def test_g02_nonindexed_vertices_remain_unwelded(self):
        doc,blob=simple_glb();corners=[0,0,0,1,0,0,1,1,0,0,0,0,1,1,0,0,1,0]
        blob=struct.pack("<18f",*corners)
        doc['buffers'][0]['byteLength']=len(blob);doc['bufferViews']=[{'buffer':0,'byteLength':len(blob)}]
        doc['accessors']=[{'bufferView':0,'componentType':5126,'type':'VEC3','count':6,'min':[0,0,0],'max':[1,1,0]}]
        doc['meshes'][0]['primitives'][0].pop('indices')
        mesh=load_bytes(pack_glb(doc,blob),'.glb').meshes[0]
        self.assertEqual(mesh.vertices.shape,(6,3));self.assertEqual(mesh.faces.tolist(),[[0,1,2],[3,4,5]])

    def test_g03_all_index_widths(self):
        for component in [5121,5123,5125]:
            doc,blob=simple_glb(component)
            self.assertEqual(load_bytes(pack_glb(doc,blob),'.glb').meshes[0].faces.tolist(),[[0,1,2],[0,2,3]])

    def test_g04_offsets_interleaving_and_sentinels(self):
        doc,old=simple_glb();positions=[[0,0,0],[1,0,0],[1,1,0],[0,1,0]]
        blob=b'\0'*16+b''.join(struct.pack('<8f',999,*p,0,0,1,777) for p in positions)+old[48:]
        doc['buffers'][0]['byteLength']=len(blob)
        doc['bufferViews']=[{'buffer':0,'byteOffset':16,'byteLength':128,'byteStride':32},{'buffer':0,'byteOffset':144,'byteLength':12}]
        doc['accessors'][0]['byteOffset']=4
        doc['accessors'].append({'bufferView':0,'byteOffset':16,'componentType':5126,'count':4,'type':'VEC3'})
        doc['meshes'][0]['primitives'][0]['attributes']['NORMAL']=2
        mesh=load_bytes(pack_glb(doc,blob),'.glb').meshes[0]
        np.testing.assert_array_equal(mesh.vertices,positions);np.testing.assert_array_equal(mesh.vertex_normals,[[0,0,1]]*4)

    def test_g05_normalized_integer_colors(self):
        for component,fmt,middle,maximum in [(5121,'B',128,255),(5123,'H',32768,65535)]:
            doc,blob=simple_glb();offset=(len(blob)+3)//4*4;blob+=b'\0'*(offset-len(blob))
            colors=struct.pack('<'+fmt*16,*([0,middle,maximum,maximum]*4));blob+=colors
            doc['buffers'][0]['byteLength']=len(blob);doc['bufferViews'].append({'buffer':0,'byteOffset':offset,'byteLength':len(colors)})
            doc['accessors'].append({'bufferView':2,'componentType':component,'type':'VEC4','count':4,'normalized':True})
            doc['meshes'][0]['primitives'][0]['attributes']['COLOR_0']=2
            mesh=load_bytes(pack_glb(doc,blob),'.glb').meshes[0]
            np.testing.assert_allclose(mesh.color_sets[0][0],[0,middle/maximum,1,1],rtol=0,atol=1e-15)

    def test_g06_zero_initialized_accessor_is_not_an_empty_asset(self):
        doc,blob=simple_glb();doc['accessors'][0].pop('bufferView');doc['accessors'][0]['max']=[0,0,0]
        mesh=load_bytes(pack_glb(doc,blob),'.glb').meshes[0]
        np.testing.assert_array_equal(mesh.vertices,np.zeros((4,3)));self.assertEqual(len(mesh.faces),2)

    def test_g07_g08_multiple_attributes_and_source_types(self):
        doc,blob=rich_glb();asset=load_bytes(pack_glb(doc,blob),'.glb');mesh=asset.meshes[0]
        self.assertEqual(sorted(mesh.uv_sets),[0,1])
        self.assertEqual(mesh.uv_sets[1].tolist(),[[.25,.5],[.5,.5],[.25,.75]])
        self.assertEqual(mesh.tangents[:,3].tolist(),[1,-1,1]);self.assertIsNone(mesh.face_normals)
        self.assertEqual(mesh.custom_attributes['_LABEL'].dtype,np.dtype('uint16'))
        self.assertEqual(mesh.custom_attributes['_LABEL'].ravel().tolist(),[11,12,13])
        self.assertEqual(mesh.custom_attributes['_VALUE'].dtype,np.dtype('float32'))
        self.assertEqual(mesh.custom_attributes['_VALUE'].tolist(),[[.25,.5],[.5,.75],[.75,1]])
        self.assertEqual(asset.meshes[0].color_sets[0].shape,(3,3));self.assertEqual(asset.meshes[1].color_sets[0].shape,(3,4))

    def test_g12_g13_shared_primitives_hierarchy_and_trs(self):
        doc,blob=rich_glb();asset=load_bytes(pack_glb(doc,blob),'.glb')
        self.assertEqual(len(asset.meshes),2);self.assertEqual(len(asset.nodes),3)
        self.assertEqual(asset.nodes[1].mesh_indices,asset.nodes[2].mesh_indices)
        self.assertEqual(asset.nodes[0].mesh_indices,[])
        world=asset.nodes[0].local_transform@asset.nodes[1].local_transform@np.array([1,0,0,1])
        np.testing.assert_allclose(world,[12,5,4,1],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(asset.meshes[0].vertices[1],[1,0,0])
        np.testing.assert_array_equal(asset.nodes[2].local_transform,[[1,0,0,-1],[0,2,0,0],[0,0,3,2],[0,0,0,1]])

    def test_g14_negative_and_zero_scale_do_not_rewrite_shared_faces(self):
        doc,blob=rich_glb();doc['nodes'][1]['scale']=[-1,0,1]
        asset=load_bytes(pack_glb(doc,blob),'.glb')
        self.assertEqual(asset.meshes[0].faces.tolist(),[[0,1,2]])
        self.assertEqual(asset.nodes[1].local_transform[2,2],1)

    def test_g15_scene_selection_excludes_other_scene_geometry(self):
        doc,blob=rich_glb()
        doc['meshes'].append({'primitives':[{'attributes':dict(doc['meshes'][0]['primitives'][0]['attributes'])}]})
        doc['nodes'].append({'mesh':1})
        doc['scenes']=[{'nodes':[3]},{'nodes':[0]}];doc['scene']=1
        asset=load_bytes(pack_glb(doc,blob),'.glb');self.assertEqual(asset.source.selected_scene_index,1)
        self.assertEqual(len(asset.meshes),2);self.assertEqual(len(asset.nodes),3)
        doc['scenes']=[{'nodes':[]}];doc['scene']=0
        self.assert_code(doc,blob,'unsupported_asset')
        doc,blob=simple_glb();doc.pop('scene');self.assertEqual(len(load_bytes(pack_glb(doc,blob),'.glb').meshes),1)

    def test_g16_degenerate_duplicate_faces_and_isolated_vertices_survive(self):
        doc,blob=simple_glb();blob=blob[:48]+struct.pack('<6H',0,0,1,0,0,1)
        mesh=load_bytes(pack_glb(doc,blob),'.glb').meshes[0]
        self.assertEqual(mesh.vertices.shape,(4,3));self.assertEqual(mesh.faces.tolist(),[[0,0,1],[0,0,1]])

    def test_g17_extras_metadata_and_unused_skin_are_not_resources(self):
        doc,blob=simple_glb();doc['nodes'][0]['extras']={'url':'https://example.org/source','extensions':{'unrelated':'metadata'}}
        doc['skins']=[{'joints':[0]}]
        asset=load_bytes(pack_glb(doc,blob),'.glb');self.assertIn('nodes',asset.source.metadata)

    def test_g18_padding_unknown_chunks_and_data_uri_buffer(self):
        doc,blob=simple_glb(5121)
        first=load_bytes(pack_glb(doc,blob,[(0xffffffff,b'xxxx')]),'.glb')
        doc['buffers'][0]['uri']='data:application/octet-stream;base64,'+base64.b64encode(blob).decode()
        second=load_bytes(pack_glb(doc),'.glb')
        np.testing.assert_array_equal(first.meshes[0].vertices,second.meshes[0].vertices)

    def test_u01_container_corruption_is_invalid(self):
        doc,blob=simple_glb();raw=pack_glb(doc,blob)
        for bad in [raw[:-1],b'nope'+raw[4:],raw[:16],raw[:12]+struct.pack('<I',999999)+raw[16:]]:
            with self.subTest(raw=bad),self.assertRaises(AssetImportError) as result: load_bytes(bad,'.glb')
            self.assertEqual(result.exception.code,'invalid_asset')

    def test_u03_unsupported_primitive_does_not_return_partial_asset(self):
        for mode in [0,1,5,6]:
            doc,blob=simple_glb();doc['meshes'][0]['primitives'].append({'attributes':{'POSITION':0},'mode':mode})
            self.assert_code(doc,blob,'unsupported_asset')

    def test_u04_sparse_and_quantization_are_explicitly_unsupported(self):
        doc,blob=simple_glb();offset=len(blob);blob+=b'\0'*4+struct.pack('<3f',0,0,0)
        doc['bufferViews'] += [{'buffer':0,'byteOffset':offset,'byteLength':1},{'buffer':0,'byteOffset':offset+4,'byteLength':12}]
        doc['buffers'][0]['byteLength']=len(blob)
        doc['accessors'][0]['sparse']={'count':1,'indices':{'bufferView':2,'componentType':5121},'values':{'bufferView':3}}
        self.assert_code(doc,blob,'unsupported_asset')
        doc,blob=simple_glb();doc['extensionsUsed']=['KHR_mesh_quantization'];doc['extensionsRequired']=['KHR_mesh_quantization']
        start=len(blob);blob+=b''.join(struct.pack('<3H2x',*p) for p in [(0,0,0),(1,0,0),(1,1,0),(0,1,0)])
        doc['bufferViews'].append({'buffer':0,'byteOffset':start,'byteLength':32,'byteStride':8})
        doc['buffers'][0]['byteLength']=len(blob);doc['accessors'][0].update(bufferView=2,componentType=5123)
        self.assert_code(doc,blob,'unsupported_asset')

    def test_u05_external_and_optional_unknown_extensions_rejected(self):
        for extension in ['VENDOR_unknown','KHR_lights_punctual','KHR_texture_basisu']:
            doc,blob=simple_glb();doc['extensionsUsed']=[extension]
            self.assert_code(doc,blob,'unsupported_asset')
        doc,blob=simple_glb();doc['buffers'][0]['uri']='external.bin';self.assert_code(doc,blob,'unsupported_asset')
        doc,blob=simple_glb();doc['images']=[{'uri':'image.png'}];self.assert_code(doc,blob,'unsupported_asset')

    def test_u07_dynamic_features_rejected_even_outside_selected_scene(self):
        for change in ['animation','skin','morph']:
            doc,blob=simple_glb()
            if change=='animation':
                start=len(blob);blob+=struct.pack('<2f6f',0,1,0,0,0,1,0,0)
                doc['bufferViews'] += [{'buffer':0,'byteOffset':start,'byteLength':8},{'buffer':0,'byteOffset':start+8,'byteLength':24}]
                doc['accessors'] += [{'bufferView':2,'componentType':5126,'count':2,'type':'SCALAR','min':[0],'max':[1]},
                                     {'bufferView':3,'componentType':5126,'count':2,'type':'VEC3'}]
                doc['animations']=[{'channels':[{'sampler':0,'target':{'node':0,'path':'translation'}}],'samplers':[{'input':2,'output':3}]}]
                doc['buffers'][0]['byteLength']=len(blob)
            elif change=='skin':
                start=len(blob);blob+=struct.pack('<16H',*([0]*16))+struct.pack('<16f',*([1,0,0,0]*4))
                doc['bufferViews'] += [{'buffer':0,'byteOffset':start,'byteLength':32},{'buffer':0,'byteOffset':start+32,'byteLength':64}]
                doc['accessors'] += [{'bufferView':2,'componentType':5123,'count':4,'type':'VEC4'},{'bufferView':3,'componentType':5126,'count':4,'type':'VEC4'}]
                doc['meshes'][0]['primitives'][0]['attributes'].update(JOINTS_0=2,WEIGHTS_0=3)
                doc['nodes'] += [{},{'mesh':0,'skin':0}];doc['skins']=[{'joints':[1]}]
                doc['scenes'].append({'nodes':[1,2]});doc['buffers'][0]['byteLength']=len(blob)
            else: doc['meshes'].append({'primitives':[{'attributes':{'POSITION':0},'indices':1,'targets':[{'POSITION':0}]}]})
            self.assert_code(doc,blob,'unsupported_asset')

    def test_u08_independent_array_and_reference_faults(self):
        modifications=[lambda d:d['nodes'][0].update(mesh=5),lambda d:d['accessors'][0].update(count=5),
                       lambda d:d['accessors'][1].update(count=4),lambda d:d['bufferViews'][0].update(byteLength=9999),
                       lambda d:d['accessors'][0].update(byteOffset=2),lambda d:d['bufferViews'][0].update(byteStride=8)]
        for modify in modifications:
            doc,blob=simple_glb();modify(doc);self.assert_code(doc,blob,'invalid_asset')
        doc,blob=simple_glb(5125);blob=blob[:-24]+struct.pack('<6I',4294967294,1,2,0,2,3)
        self.assert_code(doc,blob,'invalid_asset')

    def test_u09_nonfinite_geometry_and_attributes_rejected(self):
        doc,blob=simple_glb();blob=struct.pack('<f',float('nan'))+blob[4:];self.assert_code(doc,blob,'invalid_asset')
        doc,blob=rich_glb();normal=doc['accessors'][1];offset=doc['bufferViews'][normal['bufferView']]['byteOffset']
        blob=blob[:offset]+struct.pack('<f',float('inf'))+blob[offset+4:];self.assert_code(doc,blob,'invalid_asset')

    def test_u10_scene_and_graph_errors_are_distinguished(self):
        doc,blob=simple_glb();doc.pop('scenes');self.assert_code(doc,blob,'unsupported_asset')
        doc,blob=simple_glb();doc['scenes']=[];self.assert_code(doc,blob,'invalid_asset')
        doc,blob=simple_glb();doc['scene']=4;self.assert_code(doc,blob,'invalid_asset')
        doc,blob=simple_glb();doc['nodes'][0]['children']=[0];self.assert_code(doc,blob,'invalid_asset')
        doc,blob=simple_glb();doc['scenes']=[{},{}];doc.pop('scene');self.assert_code(doc,blob,'unsupported_asset')
        doc,blob=simple_glb();doc['accessors'][0]['count']=0;self.assert_code(doc,blob,'invalid_asset')
