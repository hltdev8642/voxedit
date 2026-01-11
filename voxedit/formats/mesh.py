"""
3D Mesh Export Formats
======================

Export voxel models to standard 3D mesh formats:
- OBJ (Wavefront)
- STL (Stereolithography)
- PLY (Polygon File Format)
- GLTF/GLB (GL Transmission Format)

Supports optimized mesh generation with:
- Greedy meshing for reduced polygon count
- Per-face UV mapping
- Vertex colors
- Material definitions
"""

import struct
import json
import base64
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field

from voxedit.core.voxel_model import VoxelModel
from voxedit.core.palette import VoxelPalette


@dataclass
class Vertex:
    """3D vertex with position, normal, color, and UV."""
    position: Tuple[float, float, float]
    normal: Tuple[float, float, float] = (0, 0, 0)
    color: Tuple[float, float, float, float] = (1, 1, 1, 1)
    uv: Tuple[float, float] = (0, 0)


@dataclass
class Face:
    """Triangle or quad face."""
    vertices: List[int]
    material_id: int = 0


@dataclass
class Mesh:
    """Collection of vertices and faces."""
    vertices: List[Vertex] = field(default_factory=list)
    faces: List[Face] = field(default_factory=list)
    materials: Dict[int, Dict[str, Any]] = field(default_factory=dict)


class VoxelMesher:
    """
    Converts voxel data to polygon mesh.
    
    Uses greedy meshing algorithm for optimized output.
    """
    
    # Face directions with normals
    FACES = {
        'right':  ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
        'left':   ((-1, 0, 0), [(0, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 0)]),
        'top':    ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
        'bottom': ((0, -1, 0), [(0, 0, 1), (0, 0, 0), (1, 0, 0), (1, 0, 1)]),
        'front':  ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
        'back':   ((0, 0, -1), [(1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0)]),
    }
    
    @classmethod
    def generate_mesh(cls, model: VoxelModel, palette: VoxelPalette,
                      scale: float = 1.0, center: bool = True,
                      optimize: bool = True) -> Mesh:
        """
        Generate a mesh from a voxel model.
        
        Args:
            model: VoxelModel to convert
            palette: Color palette for materials
            scale: Scale factor for the mesh
            center: If True, center the mesh at origin
            optimize: If True, use greedy meshing (slower but fewer polygons)
            
        Returns:
            Mesh object with vertices and faces
        """
        mesh = Mesh()
        
        # Calculate centering offset
        if center:
            offset = (-model.size[0] / 2, -model.size[1] / 2, -model.size[2] / 2)
        else:
            offset = (0, 0, 0)
        
        if optimize:
            cls._generate_greedy_mesh(model, palette, mesh, scale, offset)
        else:
            cls._generate_simple_mesh(model, palette, mesh, scale, offset)
        
        # Generate materials
        used_colors = set()
        for face in mesh.faces:
            used_colors.add(face.material_id)
        
        for color_idx in used_colors:
            if 0 < color_idx < len(palette.colors):
                c = palette.colors[color_idx]
                mesh.materials[color_idx] = {
                    'name': f'color_{color_idx}',
                    'diffuse': (c.r / 255, c.g / 255, c.b / 255),
                    'alpha': c.a / 255,
                    'metallic': c.metallic,
                    'roughness': c.roughness,
                    'emission': c.emission
                }
        
        return mesh
    
    @classmethod
    def _generate_simple_mesh(cls, model: VoxelModel, palette: VoxelPalette,
                               mesh: Mesh, scale: float, offset: Tuple[float, float, float]):
        """Generate mesh with per-voxel faces (unoptimized)."""
        for x in range(model.size[0]):
            for y in range(model.size[1]):
                for z in range(model.size[2]):
                    voxel = model.get_voxel(x, y, z)
                    if voxel == 0:
                        continue
                    
                    color = palette.get_color(voxel)
                    color_tuple = color.to_float()
                    
                    # Check each face
                    for face_name, (normal, corners) in cls.FACES.items():
                        # Check if neighbor exists
                        nx = x + normal[0]
                        ny = y + normal[1]
                        nz = z + normal[2]
                        
                        if model.is_valid_position(nx, ny, nz) and model.get_voxel(nx, ny, nz) > 0:
                            continue  # Skip hidden face
                        
                        # Add face vertices
                        base_idx = len(mesh.vertices)
                        
                        for corner in corners:
                            pos = (
                                (x + corner[0] + offset[0]) * scale,
                                (y + corner[1] + offset[1]) * scale,
                                (z + corner[2] + offset[2]) * scale
                            )
                            vertex = Vertex(
                                position=pos,
                                normal=normal,
                                color=color_tuple,
                                uv=(corner[0], corner[1])
                            )
                            mesh.vertices.append(vertex)
                        
                        # Add two triangles for quad
                        mesh.faces.append(Face(vertices=[base_idx, base_idx + 1, base_idx + 2], material_id=voxel))
                        mesh.faces.append(Face(vertices=[base_idx, base_idx + 2, base_idx + 3], material_id=voxel))
    
    @classmethod
    def _generate_greedy_mesh(cls, model: VoxelModel, palette: VoxelPalette,
                               mesh: Mesh, scale: float, offset: Tuple[float, float, float]):
        """Generate mesh using greedy meshing algorithm."""
        # Process each axis
        for axis in range(3):
            # For each slice perpendicular to axis
            for d in range(model.size[axis] + 1):
                # Create mask of visible faces
                if axis == 0:
                    slice_size = (model.size[1], model.size[2])
                elif axis == 1:
                    slice_size = (model.size[0], model.size[2])
                else:
                    slice_size = (model.size[0], model.size[1])
                
                mask = {}
                
                for u in range(slice_size[0]):
                    for v in range(slice_size[1]):
                        # Get positions for this face
                        if axis == 0:
                            pos = (d, u, v)
                            prev_pos = (d - 1, u, v)
                        elif axis == 1:
                            pos = (u, d, v)
                            prev_pos = (u, d - 1, v)
                        else:
                            pos = (u, v, d)
                            prev_pos = (u, v, d - 1)
                        
                        # Get voxels on both sides
                        current = model.get_voxel(*pos) if model.is_valid_position(*pos) else 0
                        previous = model.get_voxel(*prev_pos) if model.is_valid_position(*prev_pos) else 0
                        
                        # Check if face is visible
                        if current > 0 and previous == 0:
                            mask[(u, v)] = (current, 1)  # Face normal in positive direction
                        elif previous > 0 and current == 0:
                            mask[(u, v)] = (previous, -1)  # Face normal in negative direction
                
                # Greedy merge faces
                cls._merge_and_add_faces(mask, axis, d, slice_size, model, palette, mesh, scale, offset)
    
    @classmethod
    def _merge_and_add_faces(cls, mask: Dict, axis: int, d: int, slice_size: Tuple[int, int],
                              model: VoxelModel, palette: VoxelPalette, mesh: Mesh,
                              scale: float, offset: Tuple[float, float, float]):
        """Merge adjacent same-colored faces and add to mesh."""
        visited = set()
        
        for (u, v), (color_idx, direction) in mask.items():
            if (u, v) in visited:
                continue
            
            # Find width of merged face
            width = 1
            while (u + width, v) in mask and mask[(u + width, v)] == (color_idx, direction) and (u + width, v) not in visited:
                width += 1
            
            # Find height of merged face
            height = 1
            done = False
            while not done:
                for w in range(width):
                    if (u + w, v + height) not in mask or mask[(u + w, v + height)] != (color_idx, direction) or (u + w, v + height) in visited:
                        done = True
                        break
                if not done:
                    height += 1
            
            # Mark as visited
            for w in range(width):
                for h in range(height):
                    visited.add((u + w, v + h))
            
            # Create face
            color = palette.get_color(color_idx)
            color_tuple = color.to_float()
            
            # Calculate vertices
            if axis == 0:
                normal = (direction, 0, 0)
                if direction > 0:
                    corners = [
                        (d, u, v),
                        (d, u + width, v),
                        (d, u + width, v + height),
                        (d, u, v + height)
                    ]
                else:
                    corners = [
                        (d, u, v + height),
                        (d, u + width, v + height),
                        (d, u + width, v),
                        (d, u, v)
                    ]
            elif axis == 1:
                normal = (0, direction, 0)
                if direction > 0:
                    corners = [
                        (u, d, v),
                        (u, d, v + height),
                        (u + width, d, v + height),
                        (u + width, d, v)
                    ]
                else:
                    corners = [
                        (u + width, d, v),
                        (u + width, d, v + height),
                        (u, d, v + height),
                        (u, d, v)
                    ]
            else:
                normal = (0, 0, direction)
                if direction > 0:
                    corners = [
                        (u, v, d),
                        (u + width, v, d),
                        (u + width, v + height, d),
                        (u, v + height, d)
                    ]
                else:
                    corners = [
                        (u, v + height, d),
                        (u + width, v + height, d),
                        (u + width, v, d),
                        (u, v, d)
                    ]
            
            # Add vertices
            base_idx = len(mesh.vertices)
            for corner in corners:
                pos = (
                    (corner[0] + offset[0]) * scale,
                    (corner[1] + offset[1]) * scale,
                    (corner[2] + offset[2]) * scale
                )
                mesh.vertices.append(Vertex(position=pos, normal=normal, color=color_tuple))
            
            # Add triangles
            mesh.faces.append(Face(vertices=[base_idx, base_idx + 1, base_idx + 2], material_id=color_idx))
            mesh.faces.append(Face(vertices=[base_idx, base_idx + 2, base_idx + 3], material_id=color_idx))


class ObjExporter:
    """Export meshes to Wavefront OBJ format."""
    
    @classmethod
    def export(cls, filepath: str, model: VoxelModel, palette: VoxelPalette,
               scale: float = 1.0, optimize: bool = True):
        """
        Export a voxel model to OBJ format.
        
        Args:
            filepath: Output file path
            model: VoxelModel to export
            palette: Color palette
            scale: Scale factor
            optimize: Use greedy meshing
        """
        mesh = VoxelMesher.generate_mesh(model, palette, scale, center=True, optimize=optimize)
        
        # Write MTL file
        mtl_path = Path(filepath).with_suffix('.mtl')
        cls._write_mtl(mtl_path, mesh)
        
        # Write OBJ file
        with open(filepath, 'w') as f:
            f.write(f"# VoxEdit OBJ Export\n")
            f.write(f"# Vertices: {len(mesh.vertices)}\n")
            f.write(f"# Faces: {len(mesh.faces)}\n")
            f.write(f"mtllib {mtl_path.name}\n\n")
            
            # Write vertices
            for v in mesh.vertices:
                f.write(f"v {v.position[0]:.6f} {v.position[1]:.6f} {v.position[2]:.6f}\n")
            
            f.write("\n")
            
            # Write normals
            for v in mesh.vertices:
                f.write(f"vn {v.normal[0]:.6f} {v.normal[1]:.6f} {v.normal[2]:.6f}\n")
            
            f.write("\n")
            
            # Write vertex colors as comments (some software supports this)
            f.write("# Vertex colors (non-standard)\n")
            for v in mesh.vertices:
                f.write(f"# vc {v.color[0]:.3f} {v.color[1]:.3f} {v.color[2]:.3f}\n")
            
            f.write("\n")
            
            # Group faces by material
            faces_by_material: Dict[int, List[Face]] = {}
            for face in mesh.faces:
                if face.material_id not in faces_by_material:
                    faces_by_material[face.material_id] = []
                faces_by_material[face.material_id].append(face)
            
            # Write faces
            for material_id, faces in faces_by_material.items():
                f.write(f"\nusemtl color_{material_id}\n")
                for face in faces:
                    indices = " ".join(f"{v+1}//{v+1}" for v in face.vertices)
                    f.write(f"f {indices}\n")
    
    @classmethod
    def _write_mtl(cls, filepath: Path, mesh: Mesh):
        """Write material library file."""
        with open(filepath, 'w') as f:
            f.write("# VoxEdit Material Library\n\n")
            
            for material_id, mat in mesh.materials.items():
                f.write(f"newmtl color_{material_id}\n")
                f.write(f"Kd {mat['diffuse'][0]:.3f} {mat['diffuse'][1]:.3f} {mat['diffuse'][2]:.3f}\n")
                f.write(f"Ka 0.1 0.1 0.1\n")
                f.write(f"Ks 0.0 0.0 0.0\n")
                f.write(f"Ns 10.0\n")
                if mat['alpha'] < 1.0:
                    f.write(f"d {mat['alpha']:.3f}\n")
                f.write("\n")


class StlExporter:
    """Export meshes to STL format (binary and ASCII)."""
    
    @classmethod
    def export(cls, filepath: str, model: VoxelModel, palette: VoxelPalette,
               scale: float = 1.0, binary: bool = True, optimize: bool = True):
        """
        Export a voxel model to STL format.
        
        Args:
            filepath: Output file path
            model: VoxelModel to export
            palette: Color palette
            scale: Scale factor
            binary: If True, write binary STL; otherwise ASCII
            optimize: Use greedy meshing
        """
        mesh = VoxelMesher.generate_mesh(model, palette, scale, center=True, optimize=optimize)
        
        if binary:
            cls._write_binary(filepath, mesh)
        else:
            cls._write_ascii(filepath, mesh)
    
    @classmethod
    def _write_binary(cls, filepath: str, mesh: Mesh):
        """Write binary STL file."""
        with open(filepath, 'wb') as f:
            # Header (80 bytes)
            header = b'VoxEdit STL Export' + b'\0' * 62
            f.write(header[:80])
            
            # Triangle count
            f.write(struct.pack('<I', len(mesh.faces)))
            
            # Write triangles
            for face in mesh.faces:
                # Get vertices
                v0 = mesh.vertices[face.vertices[0]]
                v1 = mesh.vertices[face.vertices[1]]
                v2 = mesh.vertices[face.vertices[2]]
                
                # Normal
                f.write(struct.pack('<fff', *v0.normal))
                
                # Vertices
                f.write(struct.pack('<fff', *v0.position))
                f.write(struct.pack('<fff', *v1.position))
                f.write(struct.pack('<fff', *v2.position))
                
                # Attribute byte count (can encode color)
                # Use 15-bit color: 5 bits each for R, G, B
                color = v0.color
                r = int(color[0] * 31) & 0x1F
                g = int(color[1] * 31) & 0x1F
                b = int(color[2] * 31) & 0x1F
                color_attr = (1 << 15) | (b << 10) | (g << 5) | r
                f.write(struct.pack('<H', color_attr))
    
    @classmethod
    def _write_ascii(cls, filepath: str, mesh: Mesh):
        """Write ASCII STL file."""
        with open(filepath, 'w') as f:
            f.write("solid VoxEdit\n")
            
            for face in mesh.faces:
                v0 = mesh.vertices[face.vertices[0]]
                v1 = mesh.vertices[face.vertices[1]]
                v2 = mesh.vertices[face.vertices[2]]
                
                f.write(f"  facet normal {v0.normal[0]:.6f} {v0.normal[1]:.6f} {v0.normal[2]:.6f}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0.position[0]:.6f} {v0.position[1]:.6f} {v0.position[2]:.6f}\n")
                f.write(f"      vertex {v1.position[0]:.6f} {v1.position[1]:.6f} {v1.position[2]:.6f}\n")
                f.write(f"      vertex {v2.position[0]:.6f} {v2.position[1]:.6f} {v2.position[2]:.6f}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            
            f.write("endsolid VoxEdit\n")


class PlyExporter:
    """Export meshes to PLY format with vertex colors."""
    
    @classmethod
    def export(cls, filepath: str, model: VoxelModel, palette: VoxelPalette,
               scale: float = 1.0, binary: bool = True, optimize: bool = True):
        """
        Export a voxel model to PLY format.
        
        Args:
            filepath: Output file path
            model: VoxelModel to export
            palette: Color palette
            scale: Scale factor
            binary: If True, write binary PLY
            optimize: Use greedy meshing
        """
        mesh = VoxelMesher.generate_mesh(model, palette, scale, center=True, optimize=optimize)
        
        with open(filepath, 'wb' if binary else 'w') as f:
            # Write header
            header = f"""ply
format {'binary_little_endian' if binary else 'ascii'} 1.0
comment VoxEdit PLY Export
element vertex {len(mesh.vertices)}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
property uchar alpha
element face {len(mesh.faces)}
property list uchar int vertex_indices
end_header
"""
            if binary:
                f.write(header.encode('ascii'))
            else:
                f.write(header)
            
            # Write vertices
            for v in mesh.vertices:
                r = int(v.color[0] * 255)
                g = int(v.color[1] * 255)
                b = int(v.color[2] * 255)
                a = int(v.color[3] * 255)
                
                if binary:
                    f.write(struct.pack('<fff', *v.position))
                    f.write(struct.pack('<fff', *v.normal))
                    f.write(struct.pack('<BBBB', r, g, b, a))
                else:
                    f.write(f"{v.position[0]:.6f} {v.position[1]:.6f} {v.position[2]:.6f} ")
                    f.write(f"{v.normal[0]:.6f} {v.normal[1]:.6f} {v.normal[2]:.6f} ")
                    f.write(f"{r} {g} {b} {a}\n")
            
            # Write faces
            for face in mesh.faces:
                if binary:
                    f.write(struct.pack('<B', len(face.vertices)))
                    for idx in face.vertices:
                        f.write(struct.pack('<i', idx))
                else:
                    indices = ' '.join(str(i) for i in face.vertices)
                    f.write(f"{len(face.vertices)} {indices}\n")


class GltfExporter:
    """Export meshes to GLTF/GLB format."""
    
    @classmethod
    def export(cls, filepath: str, model: VoxelModel, palette: VoxelPalette,
               scale: float = 1.0, binary: bool = True, optimize: bool = True):
        """
        Export a voxel model to GLTF/GLB format.
        
        Args:
            filepath: Output file path (.gltf or .glb)
            model: VoxelModel to export
            palette: Color palette
            scale: Scale factor
            binary: If True and filepath is .gltf, embed data as base64
            optimize: Use greedy meshing
        """
        mesh = VoxelMesher.generate_mesh(model, palette, scale, center=True, optimize=optimize)
        
        # Build GLTF structure
        gltf = {
            "asset": {"version": "2.0", "generator": "VoxEdit"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": model.name}],
            "meshes": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": [],
            "materials": []
        }
        
        # Build buffer data
        buffer_data = bytearray()
        
        # Group by material
        primitives = []
        faces_by_material: Dict[int, List[Face]] = {}
        for face in mesh.faces:
            if face.material_id not in faces_by_material:
                faces_by_material[face.material_id] = []
            faces_by_material[face.material_id].append(face)
        
        # Create materials
        material_indices = {}
        for i, (material_id, mat) in enumerate(mesh.materials.items()):
            material_indices[material_id] = i
            gltf["materials"].append({
                "name": mat['name'],
                "pbrMetallicRoughness": {
                    "baseColorFactor": [mat['diffuse'][0], mat['diffuse'][1], mat['diffuse'][2], mat['alpha']],
                    "metallicFactor": mat['metallic'],
                    "roughnessFactor": mat['roughness']
                },
                "doubleSided": True
            })
        
        # Build vertex and index data for each material
        for material_id, faces in faces_by_material.items():
            # Collect vertices for this material
            vertex_map = {}
            indices = []
            positions = []
            normals = []
            colors = []
            
            for face in faces:
                for vert_idx in face.vertices:
                    if vert_idx not in vertex_map:
                        vertex_map[vert_idx] = len(positions)
                        v = mesh.vertices[vert_idx]
                        positions.extend(v.position)
                        normals.extend(v.normal)
                        colors.extend([v.color[0], v.color[1], v.color[2], v.color[3]])
                    indices.append(vertex_map[vert_idx])
            
            # Calculate bounds
            positions_array = np.array(positions).reshape(-1, 3)
            min_pos = positions_array.min(axis=0).tolist()
            max_pos = positions_array.max(axis=0).tolist()
            
            # Index buffer view
            index_offset = len(buffer_data)
            index_data = np.array(indices, dtype=np.uint32).tobytes()
            buffer_data.extend(index_data)
            # Align to 4 bytes
            while len(buffer_data) % 4 != 0:
                buffer_data.append(0)
            
            # Position buffer view
            position_offset = len(buffer_data)
            position_data = np.array(positions, dtype=np.float32).tobytes()
            buffer_data.extend(position_data)
            
            # Normal buffer view
            normal_offset = len(buffer_data)
            normal_data = np.array(normals, dtype=np.float32).tobytes()
            buffer_data.extend(normal_data)
            
            # Color buffer view
            color_offset = len(buffer_data)
            color_data = np.array(colors, dtype=np.float32).tobytes()
            buffer_data.extend(color_data)
            
            num_vertices = len(positions) // 3
            
            # Buffer views
            bv_idx = len(gltf["bufferViews"])
            gltf["bufferViews"].extend([
                {"buffer": 0, "byteOffset": index_offset, "byteLength": len(index_data), "target": 34963},
                {"buffer": 0, "byteOffset": position_offset, "byteLength": len(position_data), "target": 34962},
                {"buffer": 0, "byteOffset": normal_offset, "byteLength": len(normal_data), "target": 34962},
                {"buffer": 0, "byteOffset": color_offset, "byteLength": len(color_data), "target": 34962},
            ])
            
            # Accessors
            acc_idx = len(gltf["accessors"])
            gltf["accessors"].extend([
                {"bufferView": bv_idx, "componentType": 5125, "count": len(indices), "type": "SCALAR"},
                {"bufferView": bv_idx + 1, "componentType": 5126, "count": num_vertices, "type": "VEC3", "min": min_pos, "max": max_pos},
                {"bufferView": bv_idx + 2, "componentType": 5126, "count": num_vertices, "type": "VEC3"},
                {"bufferView": bv_idx + 3, "componentType": 5126, "count": num_vertices, "type": "VEC4"},
            ])
            
            primitives.append({
                "attributes": {
                    "POSITION": acc_idx + 1,
                    "NORMAL": acc_idx + 2,
                    "COLOR_0": acc_idx + 3
                },
                "indices": acc_idx,
                "material": material_indices.get(material_id, 0)
            })
        
        gltf["meshes"].append({"name": model.name, "primitives": primitives})
        
        # Write file
        ext = Path(filepath).suffix.lower()
        
        if ext == '.glb':
            cls._write_glb(filepath, gltf, bytes(buffer_data))
        else:
            # Write separate .bin file or embed as base64
            if binary:
                gltf["buffers"].append({
                    "byteLength": len(buffer_data),
                    "uri": f"data:application/octet-stream;base64,{base64.b64encode(buffer_data).decode('ascii')}"
                })
            else:
                bin_path = Path(filepath).with_suffix('.bin')
                with open(bin_path, 'wb') as f:
                    f.write(buffer_data)
                gltf["buffers"].append({"byteLength": len(buffer_data), "uri": bin_path.name})
            
            with open(filepath, 'w') as f:
                json.dump(gltf, f, indent=2)
    
    @classmethod
    def _write_glb(cls, filepath: str, gltf: Dict, buffer_data: bytes):
        """Write binary GLB file."""
        # Add buffer without URI
        gltf["buffers"] = [{"byteLength": len(buffer_data)}]
        
        # Convert JSON to bytes
        json_data = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
        # Pad to 4 bytes
        while len(json_data) % 4 != 0:
            json_data += b' '
        
        # Pad buffer to 4 bytes
        buffer_data = bytearray(buffer_data)
        while len(buffer_data) % 4 != 0:
            buffer_data.append(0)
        
        with open(filepath, 'wb') as f:
            # Header
            f.write(b'glTF')  # Magic
            f.write(struct.pack('<I', 2))  # Version
            total_length = 12 + 8 + len(json_data) + 8 + len(buffer_data)
            f.write(struct.pack('<I', total_length))
            
            # JSON chunk
            f.write(struct.pack('<I', len(json_data)))
            f.write(b'JSON')
            f.write(json_data)
            
            # Binary chunk
            f.write(struct.pack('<I', len(buffer_data)))
            f.write(b'BIN\x00')
            f.write(buffer_data)
