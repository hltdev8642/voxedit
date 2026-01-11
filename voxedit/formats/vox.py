"""
MagicaVoxel .vox File Format Handler
====================================

Complete implementation of MagicaVoxel VOX file format.
Supports reading and writing .vox files with full palette and material support.

VOX File Format Specification:
- VOX files use little-endian byte order
- File starts with 'VOX ' magic number and version
- Main chunk contains SIZE, XYZI, and RGBA chunks
- Supports multiple models (frames)

Compatible with:
- MagicaVoxel 0.99+
- Teardown game engine
- Other VOX-compatible tools
"""

import struct
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, BinaryIO
from dataclasses import dataclass, field
from pathlib import Path

from voxedit.core.voxel_model import VoxelModel
from voxedit.core.palette import VoxelPalette, PaletteColor


@dataclass
class VoxChunk:
    """Represents a chunk in the VOX file."""
    id: str
    content: bytes
    children: List['VoxChunk'] = field(default_factory=list)


@dataclass  
class VoxModel:
    """A single model/frame within a VOX file."""
    size: Tuple[int, int, int]
    voxels: List[Tuple[int, int, int, int]]  # x, y, z, color_index
    
    
@dataclass
class VoxScene:
    """Complete VOX scene with multiple models and palette."""
    models: List[VoxModel]
    palette: List[Tuple[int, int, int, int]]  # RGBA colors
    materials: Dict[int, Dict[str, Any]]
    transforms: List[Dict[str, Any]]
    groups: List[Dict[str, Any]]
    shapes: List[Dict[str, Any]]
    layers: List[Dict[str, Any]]


class VoxFormat:
    """
    MagicaVoxel .vox file format reader/writer.
    
    Supports:
    - Reading/writing MagicaVoxel .vox files
    - Multiple models per file
    - Full palette with 256 colors
    - Material properties (metal, glass, emit)
    - Scene hierarchy (transforms, groups, shapes)
    """
    
    MAGIC = b'VOX '
    VERSION = 150  # MagicaVoxel version 0.99
    
    # Default MagicaVoxel palette
    DEFAULT_PALETTE = [
        (0, 0, 0, 0),  # Index 0 is always empty
        (255, 255, 255, 255), (255, 255, 204, 255), (255, 255, 153, 255), (255, 255, 102, 255),
        (255, 255, 51, 255), (255, 255, 0, 255), (255, 204, 255, 255), (255, 204, 204, 255),
        (255, 204, 153, 255), (255, 204, 102, 255), (255, 204, 51, 255), (255, 204, 0, 255),
        (255, 153, 255, 255), (255, 153, 204, 255), (255, 153, 153, 255), (255, 153, 102, 255),
        (255, 153, 51, 255), (255, 153, 0, 255), (255, 102, 255, 255), (255, 102, 204, 255),
        (255, 102, 153, 255), (255, 102, 102, 255), (255, 102, 51, 255), (255, 102, 0, 255),
        (255, 51, 255, 255), (255, 51, 204, 255), (255, 51, 153, 255), (255, 51, 102, 255),
        (255, 51, 51, 255), (255, 51, 0, 255), (255, 0, 255, 255), (255, 0, 204, 255),
        (255, 0, 153, 255), (255, 0, 102, 255), (255, 0, 51, 255), (255, 0, 0, 255),
        (204, 255, 255, 255), (204, 255, 204, 255), (204, 255, 153, 255), (204, 255, 102, 255),
        (204, 255, 51, 255), (204, 255, 0, 255), (204, 204, 255, 255), (204, 204, 204, 255),
        (204, 204, 153, 255), (204, 204, 102, 255), (204, 204, 51, 255), (204, 204, 0, 255),
        (204, 153, 255, 255), (204, 153, 204, 255), (204, 153, 153, 255), (204, 153, 102, 255),
        (204, 153, 51, 255), (204, 153, 0, 255), (204, 102, 255, 255), (204, 102, 204, 255),
        (204, 102, 153, 255), (204, 102, 102, 255), (204, 102, 51, 255), (204, 102, 0, 255),
        (204, 51, 255, 255), (204, 51, 204, 255), (204, 51, 153, 255), (204, 51, 102, 255),
        (204, 51, 51, 255), (204, 51, 0, 255), (204, 0, 255, 255), (204, 0, 204, 255),
        (204, 0, 153, 255), (204, 0, 102, 255), (204, 0, 51, 255), (204, 0, 0, 255),
        (153, 255, 255, 255), (153, 255, 204, 255), (153, 255, 153, 255), (153, 255, 102, 255),
        (153, 255, 51, 255), (153, 255, 0, 255), (153, 204, 255, 255), (153, 204, 204, 255),
        (153, 204, 153, 255), (153, 204, 102, 255), (153, 204, 51, 255), (153, 204, 0, 255),
        (153, 153, 255, 255), (153, 153, 204, 255), (153, 153, 153, 255), (153, 153, 102, 255),
        (153, 153, 51, 255), (153, 153, 0, 255), (153, 102, 255, 255), (153, 102, 204, 255),
        (153, 102, 153, 255), (153, 102, 102, 255), (153, 102, 51, 255), (153, 102, 0, 255),
        (153, 51, 255, 255), (153, 51, 204, 255), (153, 51, 153, 255), (153, 51, 102, 255),
        (153, 51, 51, 255), (153, 51, 0, 255), (153, 0, 255, 255), (153, 0, 204, 255),
        (153, 0, 153, 255), (153, 0, 102, 255), (153, 0, 51, 255), (153, 0, 0, 255),
        (102, 255, 255, 255), (102, 255, 204, 255), (102, 255, 153, 255), (102, 255, 102, 255),
        (102, 255, 51, 255), (102, 255, 0, 255), (102, 204, 255, 255), (102, 204, 204, 255),
        (102, 204, 153, 255), (102, 204, 102, 255), (102, 204, 51, 255), (102, 204, 0, 255),
        (102, 153, 255, 255), (102, 153, 204, 255), (102, 153, 153, 255), (102, 153, 102, 255),
        (102, 153, 51, 255), (102, 153, 0, 255), (102, 102, 255, 255), (102, 102, 204, 255),
        (102, 102, 153, 255), (102, 102, 102, 255), (102, 102, 51, 255), (102, 102, 0, 255),
        (102, 51, 255, 255), (102, 51, 204, 255), (102, 51, 153, 255), (102, 51, 102, 255),
        (102, 51, 51, 255), (102, 51, 0, 255), (102, 0, 255, 255), (102, 0, 204, 255),
        (102, 0, 153, 255), (102, 0, 102, 255), (102, 0, 51, 255), (102, 0, 0, 255),
        (51, 255, 255, 255), (51, 255, 204, 255), (51, 255, 153, 255), (51, 255, 102, 255),
        (51, 255, 51, 255), (51, 255, 0, 255), (51, 204, 255, 255), (51, 204, 204, 255),
        (51, 204, 153, 255), (51, 204, 102, 255), (51, 204, 51, 255), (51, 204, 0, 255),
        (51, 153, 255, 255), (51, 153, 204, 255), (51, 153, 153, 255), (51, 153, 102, 255),
        (51, 153, 51, 255), (51, 153, 0, 255), (51, 102, 255, 255), (51, 102, 204, 255),
        (51, 102, 153, 255), (51, 102, 102, 255), (51, 102, 51, 255), (51, 102, 0, 255),
        (51, 51, 255, 255), (51, 51, 204, 255), (51, 51, 153, 255), (51, 51, 102, 255),
        (51, 51, 51, 255), (51, 51, 0, 255), (51, 0, 255, 255), (51, 0, 204, 255),
        (51, 0, 153, 255), (51, 0, 102, 255), (51, 0, 51, 255), (51, 0, 0, 255),
        (0, 255, 255, 255), (0, 255, 204, 255), (0, 255, 153, 255), (0, 255, 102, 255),
        (0, 255, 51, 255), (0, 255, 0, 255), (0, 204, 255, 255), (0, 204, 204, 255),
        (0, 204, 153, 255), (0, 204, 102, 255), (0, 204, 51, 255), (0, 204, 0, 255),
        (0, 153, 255, 255), (0, 153, 204, 255), (0, 153, 153, 255), (0, 153, 102, 255),
        (0, 153, 51, 255), (0, 153, 0, 255), (0, 102, 255, 255), (0, 102, 204, 255),
        (0, 102, 153, 255), (0, 102, 102, 255), (0, 102, 51, 255), (0, 102, 0, 255),
        (0, 51, 255, 255), (0, 51, 204, 255), (0, 51, 153, 255), (0, 51, 102, 255),
        (0, 51, 51, 255), (0, 51, 0, 255), (0, 0, 255, 255), (0, 0, 204, 255),
        (0, 0, 153, 255), (0, 0, 102, 255), (0, 0, 51, 255), (238, 0, 0, 255),
        (221, 0, 0, 255), (187, 0, 0, 255), (170, 0, 0, 255), (136, 0, 0, 255),
        (119, 0, 0, 255), (85, 0, 0, 255), (68, 0, 0, 255), (34, 0, 0, 255),
        (17, 0, 0, 255), (0, 238, 0, 255), (0, 221, 0, 255), (0, 187, 0, 255),
        (0, 170, 0, 255), (0, 136, 0, 255), (0, 119, 0, 255), (0, 85, 0, 255),
        (0, 68, 0, 255), (0, 34, 0, 255), (0, 17, 0, 255), (0, 0, 238, 255),
        (0, 0, 221, 255), (0, 0, 187, 255), (0, 0, 170, 255), (0, 0, 136, 255),
        (0, 0, 119, 255), (0, 0, 85, 255), (0, 0, 68, 255), (0, 0, 34, 255),
        (0, 0, 17, 255), (238, 238, 238, 255), (221, 221, 221, 255), (187, 187, 187, 255),
        (170, 170, 170, 255), (136, 136, 136, 255), (119, 119, 119, 255), (85, 85, 85, 255),
        (68, 68, 68, 255), (34, 34, 34, 255), (17, 17, 17, 255), (0, 0, 0, 255),
    ]
    
    def __init__(self):
        """Initialize the VOX format handler."""
        self.models: List[VoxModel] = []
        self.palette: List[Tuple[int, int, int, int]] = list(self.DEFAULT_PALETTE)
        self.materials: Dict[int, Dict[str, Any]] = {}
        self._current_model_index = 0
    
    @classmethod
    def load(cls, filepath: str) -> Tuple[VoxelModel, VoxelPalette]:
        """
        Load a VOX file and return the first model with palette.
        
        Args:
            filepath: Path to the VOX file
            
        Returns:
            Tuple of (VoxelModel, VoxelPalette)
        """
        handler = cls()
        scene = handler.read(filepath)
        
        if not scene.models:
            raise ValueError("VOX file contains no models")
        
        # Convert first model to VoxelModel
        model_data = scene.models[0]
        model = VoxelModel.create(
            size_x=model_data.size[0],
            size_y=model_data.size[2],  # VOX uses Z for height
            size_z=model_data.size[1],
            name=Path(filepath).stem
        )
        
        # Set voxels
        for x, y, z, color_idx in model_data.voxels:
            # Convert VOX coordinate system to our system
            model.set_voxel(x, z, y, color_idx)
        
        model.commit()
        
        # Convert palette
        palette = VoxelPalette()
        palette.colors = []
        for r, g, b, a in scene.palette:
            palette.colors.append(PaletteColor(r=r, g=g, b=b, a=a))
        
        # Ensure 256 colors
        while len(palette.colors) < 256:
            palette.colors.append(PaletteColor())
        
        return model, palette
    
    @classmethod
    def load_all_models(cls, filepath: str) -> Tuple[List[VoxelModel], VoxelPalette]:
        """
        Load all models from a VOX file.
        
        Args:
            filepath: Path to the VOX file
            
        Returns:
            Tuple of (list of VoxelModels, VoxelPalette)
        """
        handler = cls()
        scene = handler.read(filepath)
        
        models = []
        for i, model_data in enumerate(scene.models):
            model = VoxelModel.create(
                size_x=model_data.size[0],
                size_y=model_data.size[2],
                size_z=model_data.size[1],
                name=f"{Path(filepath).stem}_{i}"
            )
            
            for x, y, z, color_idx in model_data.voxels:
                model.set_voxel(x, z, y, color_idx)
            
            model.commit()
            models.append(model)
        
        # Convert palette
        palette = VoxelPalette()
        palette.colors = []
        for r, g, b, a in scene.palette:
            palette.colors.append(PaletteColor(r=r, g=g, b=b, a=a))
        
        while len(palette.colors) < 256:
            palette.colors.append(PaletteColor())
        
        return models, palette
    
    @classmethod
    def save(cls, filepath: str, model: VoxelModel, palette: VoxelPalette):
        """
        Save a VoxelModel to a VOX file.
        
        Args:
            filepath: Output file path
            model: VoxelModel to save
            palette: Color palette to use
        """
        handler = cls()
        
        # Convert model to VOX format
        vox_model = VoxModel(
            size=(model.size[0], model.size[2], model.size[1]),  # Convert coordinate system
            voxels=[]
        )
        
        # Get all non-empty voxels
        for x, y, z, color_idx in model.get_all_voxels():
            # Convert coordinate system
            vox_model.voxels.append((x, z, y, color_idx))
        
        handler.models = [vox_model]
        
        # Convert palette
        handler.palette = [(0, 0, 0, 0)]  # Index 0 is empty
        for i in range(1, 256):
            if i < len(palette.colors):
                c = palette.colors[i]
                handler.palette.append((c.r, c.g, c.b, c.a))
            else:
                handler.palette.append((255, 255, 255, 255))
        
        handler.write(filepath)
    
    @classmethod
    def save_multi(cls, filepath: str, models: List[VoxelModel], palette: VoxelPalette):
        """
        Save multiple VoxelModels to a single VOX file.
        
        Args:
            filepath: Output file path
            models: List of VoxelModels to save
            palette: Color palette to use
        """
        handler = cls()
        
        for model in models:
            vox_model = VoxModel(
                size=(model.size[0], model.size[2], model.size[1]),
                voxels=[]
            )
            
            for x, y, z, color_idx in model.get_all_voxels():
                vox_model.voxels.append((x, z, y, color_idx))
            
            handler.models.append(vox_model)
        
        # Convert palette
        handler.palette = [(0, 0, 0, 0)]
        for i in range(1, 256):
            if i < len(palette.colors):
                c = palette.colors[i]
                handler.palette.append((c.r, c.g, c.b, c.a))
            else:
                handler.palette.append((255, 255, 255, 255))
        
        handler.write(filepath)
    
    def read(self, filepath: str) -> VoxScene:
        """
        Read a VOX file and return the complete scene.
        
        Args:
            filepath: Path to the VOX file
            
        Returns:
            VoxScene containing all models and metadata
        """
        with open(filepath, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic != self.MAGIC:
                raise ValueError(f"Invalid VOX file: expected 'VOX ', got '{magic}'")
            
            version = struct.unpack('<I', f.read(4))[0]
            
            # Read main chunk
            main_chunk = self._read_chunk(f)
            if main_chunk.id != 'MAIN':
                raise ValueError(f"Expected MAIN chunk, got '{main_chunk.id}'")
            
            # Parse chunks
            return self._parse_chunks(main_chunk.children)
    
    def _read_chunk(self, f: BinaryIO) -> VoxChunk:
        """Read a single chunk from the file."""
        chunk_id = f.read(4).decode('ascii')
        content_size = struct.unpack('<I', f.read(4))[0]
        children_size = struct.unpack('<I', f.read(4))[0]
        
        content = f.read(content_size)
        
        # Read children
        children = []
        children_end = f.tell() + children_size
        while f.tell() < children_end:
            children.append(self._read_chunk(f))
        
        return VoxChunk(id=chunk_id, content=content, children=children)
    
    def _parse_chunks(self, chunks: List[VoxChunk]) -> VoxScene:
        """Parse chunk list into a VoxScene."""
        models = []
        palette = list(self.DEFAULT_PALETTE)
        materials = {}
        transforms = []
        groups = []
        shapes = []
        layers = []
        
        current_size = (0, 0, 0)
        
        for chunk in chunks:
            if chunk.id == 'SIZE':
                x, y, z = struct.unpack('<III', chunk.content[:12])
                current_size = (x, y, z)
                
            elif chunk.id == 'XYZI':
                num_voxels = struct.unpack('<I', chunk.content[:4])[0]
                voxels = []
                
                for i in range(num_voxels):
                    offset = 4 + i * 4
                    x, y, z, c = struct.unpack('<BBBB', chunk.content[offset:offset+4])
                    voxels.append((x, y, z, c))
                
                models.append(VoxModel(size=current_size, voxels=voxels))
                
            elif chunk.id == 'RGBA':
                # Read 256 colors (255 actual colors + 1 unused at the end)
                palette = [(0, 0, 0, 0)]  # Index 0 is always empty
                for i in range(255):
                    offset = i * 4
                    r, g, b, a = struct.unpack('<BBBB', chunk.content[offset:offset+4])
                    palette.append((r, g, b, a))
                
            elif chunk.id == 'MATL':
                # Material chunk
                mat_id = struct.unpack('<I', chunk.content[:4])[0]
                properties = self._parse_dict(chunk.content[4:])
                materials[mat_id] = properties
                
            elif chunk.id == 'nTRN':
                # Transform node
                transforms.append(self._parse_transform(chunk.content))
                
            elif chunk.id == 'nGRP':
                # Group node
                groups.append(self._parse_group(chunk.content))
                
            elif chunk.id == 'nSHP':
                # Shape node
                shapes.append(self._parse_shape(chunk.content))
                
            elif chunk.id == 'LAYR':
                # Layer
                layers.append(self._parse_layer(chunk.content))
        
        return VoxScene(
            models=models,
            palette=palette,
            materials=materials,
            transforms=transforms,
            groups=groups,
            shapes=shapes,
            layers=layers
        )
    
    def _parse_dict(self, data: bytes) -> Dict[str, str]:
        """Parse a DICT structure from chunk content."""
        result = {}
        offset = 0
        num_pairs = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        for _ in range(num_pairs):
            # Key
            key_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            key = data[offset:offset+key_len].decode('utf-8')
            offset += key_len
            
            # Value
            val_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            value = data[offset:offset+val_len].decode('utf-8')
            offset += val_len
            
            result[key] = value
        
        return result
    
    def _parse_transform(self, data: bytes) -> Dict[str, Any]:
        """Parse a transform node."""
        offset = 0
        node_id = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        attributes = self._parse_dict(data[offset:])
        
        return {'node_id': node_id, 'attributes': attributes}
    
    def _parse_group(self, data: bytes) -> Dict[str, Any]:
        """Parse a group node."""
        offset = 0
        node_id = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        return {'node_id': node_id}
    
    def _parse_shape(self, data: bytes) -> Dict[str, Any]:
        """Parse a shape node."""
        offset = 0
        node_id = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        return {'node_id': node_id}
    
    def _parse_layer(self, data: bytes) -> Dict[str, Any]:
        """Parse a layer."""
        offset = 0
        layer_id = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        attributes = self._parse_dict(data[offset:])
        
        return {'layer_id': layer_id, 'attributes': attributes}
    
    def write(self, filepath: str):
        """
        Write models and palette to a VOX file.
        
        Args:
            filepath: Output file path
        """
        with open(filepath, 'wb') as f:
            # Write header
            f.write(self.MAGIC)
            f.write(struct.pack('<I', self.VERSION))
            
            # Build chunks
            chunks = []
            
            # Add SIZE and XYZI for each model
            for model in self.models:
                # SIZE chunk
                size_content = struct.pack('<III', model.size[0], model.size[1], model.size[2])
                chunks.append(VoxChunk(id='SIZE', content=size_content))
                
                # XYZI chunk
                xyzi_content = struct.pack('<I', len(model.voxels))
                for x, y, z, c in model.voxels:
                    xyzi_content += struct.pack('<BBBB', x, y, z, c)
                chunks.append(VoxChunk(id='XYZI', content=xyzi_content))
            
            # RGBA chunk (palette)
            rgba_content = b''
            for i in range(1, 256):  # Skip index 0
                if i < len(self.palette):
                    r, g, b, a = self.palette[i]
                else:
                    r, g, b, a = 255, 255, 255, 255
                rgba_content += struct.pack('<BBBB', r, g, b, a)
            # Last color (unused but required)
            rgba_content += struct.pack('<BBBB', 0, 0, 0, 0)
            chunks.append(VoxChunk(id='RGBA', content=rgba_content))
            
            # Calculate MAIN chunk sizes
            children_data = b''
            for chunk in chunks:
                children_data += self._write_chunk(chunk)
            
            # Write MAIN chunk
            f.write(b'MAIN')
            f.write(struct.pack('<I', 0))  # No content
            f.write(struct.pack('<I', len(children_data)))
            f.write(children_data)
    
    def _write_chunk(self, chunk: VoxChunk) -> bytes:
        """Write a chunk to bytes."""
        children_data = b''
        for child in chunk.children:
            children_data += self._write_chunk(child)
        
        result = chunk.id.encode('ascii')
        result += struct.pack('<I', len(chunk.content))
        result += struct.pack('<I', len(children_data))
        result += chunk.content
        result += children_data
        
        return result
    
    @staticmethod
    def get_default_palette() -> VoxelPalette:
        """Get the default MagicaVoxel palette."""
        palette = VoxelPalette()
        palette.colors = []
        
        for r, g, b, a in VoxFormat.DEFAULT_PALETTE:
            palette.colors.append(PaletteColor(r=r, g=g, b=b, a=a))
        
        # Ensure 256 colors
        while len(palette.colors) < 256:
            palette.colors.append(PaletteColor())
        
        return palette
