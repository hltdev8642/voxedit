"""
Additional Voxel Format Handlers
================================

Support for various voxel file formats:
- Binvox (.binvox) - Binary voxel format
- KVX (.kvx) - Ken Silverman's voxel format
- Qubicle (.qb, .qbt) - Qubicle voxel format
- VXL (.vxl) - Voxlap/Command & Conquer voxel format
"""

import struct
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import gzip
import zlib

from voxedit.core.voxel_model import VoxelModel
from voxedit.core.palette import VoxelPalette, PaletteColor


class BinvoxFormat:
    """
    Handler for Binvox (.binvox) voxel format.
    
    Binary voxel format commonly used in 3D machine learning.
    Stores occupancy data with optional run-length encoding.
    """
    
    @classmethod
    def load(cls, filepath: str) -> Tuple[VoxelModel, VoxelPalette]:
        """
        Load a Binvox file.
        
        Args:
            filepath: Path to the .binvox file
            
        Returns:
            Tuple of (VoxelModel, VoxelPalette)
        """
        with open(filepath, 'rb') as f:
            # Read header
            line = f.readline().decode('ascii').strip()
            if line != '#binvox 1':
                raise ValueError(f"Invalid binvox header: {line}")
            
            dims = None
            translate = (0, 0, 0)
            scale = 1.0
            
            while True:
                line = f.readline().decode('ascii').strip()
                if line.startswith('dim'):
                    parts = line.split()
                    dims = (int(parts[1]), int(parts[2]), int(parts[3]))
                elif line.startswith('translate'):
                    parts = line.split()
                    translate = (float(parts[1]), float(parts[2]), float(parts[3]))
                elif line.startswith('scale'):
                    scale = float(line.split()[1])
                elif line == 'data':
                    break
            
            if dims is None:
                raise ValueError("No dimensions found in binvox file")
            
            # Read voxel data (run-length encoded)
            data = f.read()
            
        # Decode RLE data
        voxels = np.zeros(dims[0] * dims[1] * dims[2], dtype=np.uint8)
        index = 0
        i = 0
        
        while i < len(data) - 1:
            value = data[i]
            count = data[i + 1]
            voxels[index:index + count] = 1 if value else 0
            index += count
            i += 2
        
        # Reshape to 3D (note: binvox uses depth-major order)
        voxels_3d = voxels.reshape(dims).transpose(0, 2, 1)
        
        # Create model
        model = VoxelModel.from_array(voxels_3d, name=Path(filepath).stem)
        
        # Create simple palette (binvox is binary, no colors)
        palette = VoxelPalette()
        palette.colors[1] = PaletteColor(r=128, g=128, b=128)  # Gray for filled
        
        return model, palette
    
    @classmethod
    def save(cls, filepath: str, model: VoxelModel, 
             palette: VoxelPalette = None, threshold: int = 1):
        """
        Save a VoxelModel to Binvox format.
        
        Args:
            filepath: Output file path
            model: VoxelModel to save
            palette: Unused (binvox is binary)
            threshold: Voxel values >= threshold are considered filled
        """
        dims = model.size
        
        # Convert to binary occupancy
        voxels = (model.voxels >= threshold).astype(np.uint8)
        
        # Transpose and flatten
        voxels_flat = voxels.transpose(0, 2, 1).flatten()
        
        # Run-length encode
        rle_data = bytearray()
        i = 0
        while i < len(voxels_flat):
            value = voxels_flat[i]
            count = 1
            while i + count < len(voxels_flat) and voxels_flat[i + count] == value and count < 255:
                count += 1
            rle_data.append(value)
            rle_data.append(count)
            i += count
        
        with open(filepath, 'wb') as f:
            # Write header
            f.write(b'#binvox 1\n')
            f.write(f'dim {dims[0]} {dims[1]} {dims[2]}\n'.encode('ascii'))
            f.write(b'translate 0 0 0\n')
            f.write(b'scale 1\n')
            f.write(b'data\n')
            f.write(rle_data)


class KvxFormat:
    """
    Handler for Ken Silverman's KVX voxel format.
    
    Used in Build engine games and various voxel applications.
    """
    
    @classmethod
    def load(cls, filepath: str) -> Tuple[VoxelModel, VoxelPalette]:
        """
        Load a KVX file.
        
        Args:
            filepath: Path to the .kvx file
            
        Returns:
            Tuple of (VoxelModel, VoxelPalette)
        """
        with open(filepath, 'rb') as f:
            # Read header
            numbytes = struct.unpack('<I', f.read(4))[0]
            xsiz, ysiz, zsiz = struct.unpack('<III', f.read(12))
            xpivot, ypivot, zpivot = struct.unpack('<iii', f.read(12))
            
            # Read x offsets
            xoffset = []
            for _ in range(xsiz + 1):
                xoffset.append(struct.unpack('<I', f.read(4))[0])
            
            # Read xy offsets
            xyoffset = []
            for _ in range(xsiz):
                row = []
                for _ in range(ysiz + 1):
                    row.append(struct.unpack('<H', f.read(2))[0])
                xyoffset.append(row)
            
            # Read voxel data
            voxel_data = f.read()
            
            # Read palette (at end of file)
            f.seek(-768, 2)  # 256 * 3 bytes from end
            palette_data = f.read(768)
        
        # Create model
        model = VoxelModel.create(xsiz, zsiz, ysiz, name=Path(filepath).stem)
        
        # Parse voxel data
        offset = 0
        for x in range(xsiz):
            for y in range(ysiz):
                while offset < len(voxel_data):
                    ztop = voxel_data[offset]
                    zleng = voxel_data[offset + 1]
                    cull = voxel_data[offset + 2]
                    offset += 3
                    
                    if zleng == 0:
                        break
                    
                    for z in range(zleng):
                        if offset < len(voxel_data):
                            color = voxel_data[offset]
                            offset += 1
                            
                            vz = ztop + z
                            if model.is_valid_position(x, vz, y):
                                model.set_voxel(x, vz, y, color + 1)
        
        # Create palette
        palette = VoxelPalette()
        palette.colors = [PaletteColor(r=0, g=0, b=0, a=0)]
        
        for i in range(256):
            if i * 3 + 2 < len(palette_data):
                r = palette_data[i * 3] * 4  # KVX uses 6-bit color
                g = palette_data[i * 3 + 1] * 4
                b = palette_data[i * 3 + 2] * 4
                palette.colors.append(PaletteColor(r=min(255, r), g=min(255, g), b=min(255, b)))
        
        model.commit()
        return model, palette
    
    @classmethod
    def save(cls, filepath: str, model: VoxelModel, palette: VoxelPalette):
        """
        Save a VoxelModel to KVX format.
        
        Args:
            filepath: Output file path
            model: VoxelModel to save
            palette: Color palette
        """
        xsiz, zsiz, ysiz = model.size[0], model.size[1], model.size[2]
        
        # Build column data
        columns_data = bytearray()
        xoffset = [0]
        xyoffset = []
        
        for x in range(xsiz):
            xyoffset.append([0])
            for y in range(ysiz):
                # Find voxel spans in this column
                z = 0
                while z < zsiz:
                    # Find start of span
                    while z < zsiz and model.get_voxel(x, z, y) == 0:
                        z += 1
                    
                    if z >= zsiz:
                        break
                    
                    ztop = z
                    colors = []
                    
                    # Collect span
                    while z < zsiz and model.get_voxel(x, z, y) > 0:
                        colors.append(model.get_voxel(x, z, y) - 1)
                        z += 1
                    
                    # Write span
                    columns_data.append(ztop)
                    columns_data.append(len(colors))
                    columns_data.append(0)  # cull flags
                    columns_data.extend(colors)
                
                # End marker
                columns_data.extend([0, 0, 0])
                xyoffset[x].append(len(columns_data))
            
            xoffset.append(len(columns_data))
        
        # Write file
        with open(filepath, 'wb') as f:
            # Header
            numbytes = 24 + (xsiz + 1) * 4 + xsiz * (ysiz + 1) * 2 + len(columns_data) + 768
            f.write(struct.pack('<I', numbytes))
            f.write(struct.pack('<III', xsiz, ysiz, zsiz))
            f.write(struct.pack('<iii', xsiz // 2, ysiz // 2, zsiz // 2))  # Pivot
            
            # X offsets
            for off in xoffset:
                f.write(struct.pack('<I', off))
            
            # XY offsets
            for row in xyoffset:
                for off in row:
                    f.write(struct.pack('<H', off))
            
            # Column data
            f.write(columns_data)
            
            # Palette
            for i in range(256):
                if i + 1 < len(palette.colors):
                    c = palette.colors[i + 1]
                    f.write(bytes([c.r // 4, c.g // 4, c.b // 4]))
                else:
                    f.write(bytes([0, 0, 0]))


class QubicleFormat:
    """
    Handler for Qubicle voxel formats (.qb, .qbt).
    
    Qubicle is a professional voxel editor with its own formats.
    """
    
    VERSION = 0x00000101  # 1.1.0.0
    
    @classmethod
    def load(cls, filepath: str) -> Tuple[List[VoxelModel], VoxelPalette]:
        """
        Load a Qubicle Binary file.
        
        Args:
            filepath: Path to the .qb file
            
        Returns:
            Tuple of (list of VoxelModels, VoxelPalette)
        """
        ext = Path(filepath).suffix.lower()
        
        if ext == '.qb':
            return cls._load_qb(filepath)
        elif ext == '.qbt':
            return cls._load_qbt(filepath)
        else:
            raise ValueError(f"Unsupported Qubicle format: {ext}")
    
    @classmethod
    def _load_qb(cls, filepath: str) -> Tuple[List[VoxelModel], VoxelPalette]:
        """Load Qubicle Binary (.qb) format."""
        with open(filepath, 'rb') as f:
            # Header
            version = struct.unpack('<I', f.read(4))[0]
            color_format = struct.unpack('<I', f.read(4))[0]  # 0 = RGBA, 1 = BGRA
            z_axis = struct.unpack('<I', f.read(4))[0]  # 0 = left-handed, 1 = right-handed
            compressed = struct.unpack('<I', f.read(4))[0]
            vis_mask = struct.unpack('<I', f.read(4))[0]
            num_matrices = struct.unpack('<I', f.read(4))[0]
            
            models = []
            palette = VoxelPalette()
            color_to_idx = {(0, 0, 0, 0): 0}
            next_idx = 1
            
            for _ in range(num_matrices):
                # Matrix name
                name_len = struct.unpack('<B', f.read(1))[0]
                name = f.read(name_len).decode('utf-8')
                
                # Size
                size_x = struct.unpack('<I', f.read(4))[0]
                size_y = struct.unpack('<I', f.read(4))[0]
                size_z = struct.unpack('<I', f.read(4))[0]
                
                # Position
                pos_x = struct.unpack('<i', f.read(4))[0]
                pos_y = struct.unpack('<i', f.read(4))[0]
                pos_z = struct.unpack('<i', f.read(4))[0]
                
                model = VoxelModel.create(size_x, size_z, size_y, name=name)
                model.position = (pos_x, pos_z, pos_y)
                
                if compressed:
                    # RLE compressed
                    z = 0
                    while z < size_z:
                        index = 0
                        while True:
                            data = struct.unpack('<I', f.read(4))[0]
                            
                            if data == 6:  # NEXTSLICEFLAG
                                break
                            elif data == 2:  # CODEFLAG
                                count = struct.unpack('<I', f.read(4))[0]
                                color_data = struct.unpack('<I', f.read(4))[0]
                                
                                for _ in range(count):
                                    if color_data != 0:
                                        if color_format == 0:  # RGBA
                                            r = (color_data >> 0) & 0xFF
                                            g = (color_data >> 8) & 0xFF
                                            b = (color_data >> 16) & 0xFF
                                            a = (color_data >> 24) & 0xFF
                                        else:  # BGRA
                                            b = (color_data >> 0) & 0xFF
                                            g = (color_data >> 8) & 0xFF
                                            r = (color_data >> 16) & 0xFF
                                            a = (color_data >> 24) & 0xFF
                                        
                                        color_tuple = (r, g, b, a)
                                        if color_tuple not in color_to_idx and next_idx < 256:
                                            color_to_idx[color_tuple] = next_idx
                                            palette.colors.append(PaletteColor(r=r, g=g, b=b, a=a))
                                            next_idx += 1
                                        
                                        x = index % size_x
                                        y = index // size_x
                                        color_idx = color_to_idx.get(color_tuple, 1)
                                        
                                        if z_axis == 0:
                                            model.set_voxel(x, z, y, color_idx)
                                        else:
                                            model.set_voxel(x, size_z - 1 - z, y, color_idx)
                                    
                                    index += 1
                            else:
                                # Single voxel
                                if data != 0:
                                    if color_format == 0:
                                        r = (data >> 0) & 0xFF
                                        g = (data >> 8) & 0xFF
                                        b = (data >> 16) & 0xFF
                                        a = (data >> 24) & 0xFF
                                    else:
                                        b = (data >> 0) & 0xFF
                                        g = (data >> 8) & 0xFF
                                        r = (data >> 16) & 0xFF
                                        a = (data >> 24) & 0xFF
                                    
                                    color_tuple = (r, g, b, a)
                                    if color_tuple not in color_to_idx and next_idx < 256:
                                        color_to_idx[color_tuple] = next_idx
                                        palette.colors.append(PaletteColor(r=r, g=g, b=b, a=a))
                                        next_idx += 1
                                    
                                    x = index % size_x
                                    y = index // size_x
                                    color_idx = color_to_idx.get(color_tuple, 1)
                                    
                                    if z_axis == 0:
                                        model.set_voxel(x, z, y, color_idx)
                                    else:
                                        model.set_voxel(x, size_z - 1 - z, y, color_idx)
                                
                                index += 1
                        z += 1
                else:
                    # Uncompressed
                    for z in range(size_z):
                        for y in range(size_y):
                            for x in range(size_x):
                                data = struct.unpack('<I', f.read(4))[0]
                                
                                if data != 0:
                                    if color_format == 0:
                                        r = (data >> 0) & 0xFF
                                        g = (data >> 8) & 0xFF
                                        b = (data >> 16) & 0xFF
                                        a = (data >> 24) & 0xFF
                                    else:
                                        b = (data >> 0) & 0xFF
                                        g = (data >> 8) & 0xFF
                                        r = (data >> 16) & 0xFF
                                        a = (data >> 24) & 0xFF
                                    
                                    color_tuple = (r, g, b, a)
                                    if color_tuple not in color_to_idx and next_idx < 256:
                                        color_to_idx[color_tuple] = next_idx
                                        palette.colors.append(PaletteColor(r=r, g=g, b=b, a=a))
                                        next_idx += 1
                                    
                                    color_idx = color_to_idx.get(color_tuple, 1)
                                    
                                    if z_axis == 0:
                                        model.set_voxel(x, z, y, color_idx)
                                    else:
                                        model.set_voxel(x, size_z - 1 - z, y, color_idx)
                
                model.commit()
                models.append(model)
        
        # Ensure palette has 256 colors
        while len(palette.colors) < 256:
            palette.colors.append(PaletteColor())
        
        return models, palette
    
    @classmethod
    def _load_qbt(cls, filepath: str) -> Tuple[List[VoxelModel], VoxelPalette]:
        """Load Qubicle Binary Tree (.qbt) format."""
        # QBT is more complex, using a tree structure
        # For now, return empty - full implementation would be extensive
        raise NotImplementedError("QBT format loading not yet implemented")
    
    @classmethod
    def save(cls, filepath: str, models: List[VoxelModel], palette: VoxelPalette,
             compressed: bool = True):
        """
        Save VoxelModels to Qubicle Binary format.
        
        Args:
            filepath: Output file path
            models: List of VoxelModels to save
            palette: Color palette
            compressed: Use RLE compression
        """
        with open(filepath, 'wb') as f:
            # Header
            f.write(struct.pack('<I', cls.VERSION))
            f.write(struct.pack('<I', 0))  # RGBA format
            f.write(struct.pack('<I', 1))  # Right-handed
            f.write(struct.pack('<I', 1 if compressed else 0))
            f.write(struct.pack('<I', 0))  # No visibility mask encoding
            f.write(struct.pack('<I', len(models)))
            
            for model in models:
                # Name
                name_bytes = model.name.encode('utf-8')[:255]
                f.write(struct.pack('<B', len(name_bytes)))
                f.write(name_bytes)
                
                # Size (swap Y and Z)
                f.write(struct.pack('<I', model.size[0]))
                f.write(struct.pack('<I', model.size[2]))
                f.write(struct.pack('<I', model.size[1]))
                
                # Position
                f.write(struct.pack('<i', model.position[0]))
                f.write(struct.pack('<i', model.position[2]))
                f.write(struct.pack('<i', model.position[1]))
                
                if compressed:
                    # RLE compressed
                    for z in range(model.size[1]):
                        slice_data = []
                        
                        for y in range(model.size[2]):
                            for x in range(model.size[0]):
                                voxel = model.get_voxel(x, model.size[1] - 1 - z, y)
                                
                                if voxel > 0 and voxel < len(palette.colors):
                                    c = palette.colors[voxel]
                                    color = (c.r) | (c.g << 8) | (c.b << 16) | (c.a << 24)
                                else:
                                    color = 0
                                
                                slice_data.append(color)
                        
                        # Simple RLE
                        i = 0
                        while i < len(slice_data):
                            color = slice_data[i]
                            count = 1
                            while i + count < len(slice_data) and slice_data[i + count] == color and count < 65535:
                                count += 1
                            
                            if count > 1:
                                f.write(struct.pack('<I', 2))  # CODEFLAG
                                f.write(struct.pack('<I', count))
                                f.write(struct.pack('<I', color))
                            else:
                                f.write(struct.pack('<I', color))
                            
                            i += count
                        
                        f.write(struct.pack('<I', 6))  # NEXTSLICEFLAG
                else:
                    # Uncompressed
                    for z in range(model.size[1]):
                        for y in range(model.size[2]):
                            for x in range(model.size[0]):
                                voxel = model.get_voxel(x, model.size[1] - 1 - z, y)
                                
                                if voxel > 0 and voxel < len(palette.colors):
                                    c = palette.colors[voxel]
                                    color = (c.r) | (c.g << 8) | (c.b << 16) | (c.a << 24)
                                else:
                                    color = 0
                                
                                f.write(struct.pack('<I', color))


class VxlFormat:
    """
    Handler for VXL voxel format.
    
    Used in Voxlap engine and Command & Conquer games.
    """
    
    @classmethod
    def load(cls, filepath: str) -> Tuple[VoxelModel, VoxelPalette]:
        """
        Load a VXL file.
        
        Args:
            filepath: Path to the .vxl file
            
        Returns:
            Tuple of (VoxelModel, VoxelPalette)
        """
        with open(filepath, 'rb') as f:
            # Read header
            magic = f.read(16).decode('ascii').strip('\x00')
            if not magic.startswith('Voxel Animation'):
                raise ValueError(f"Invalid VXL file")
            
            palette_count = struct.unpack('<I', f.read(4))[0]
            header_count = struct.unpack('<I', f.read(4))[0]
            tailer_count = struct.unpack('<I', f.read(4))[0]
            body_size = struct.unpack('<I', f.read(4))[0]
            
            # Skip to palette
            f.seek(802)
            palette_data = f.read(256 * 3)
            
            # Read first limb header
            f.seek(802 + 256 * 3)
            
            limb_name = f.read(16).decode('ascii').strip('\x00')
            limb_num = struct.unpack('<I', f.read(4))[0]
            f.read(4)  # unknown
            f.read(4)  # unknown
            
            size_x, size_y, size_z = struct.unpack('<III', f.read(12))
            normal_type = struct.unpack('<B', f.read(1))[0]
            f.read(3)  # padding
            
            # Skip to span data
            f.seek(802 + 256 * 3 + 32)
            
            # This is a simplified loader - full VXL support is complex
            model = VoxelModel.create(size_x, size_z, size_y, name=Path(filepath).stem)
            
            # Create palette
            palette = VoxelPalette()
            palette.colors = [PaletteColor(r=0, g=0, b=0, a=0)]
            
            for i in range(255):
                if i * 3 + 2 < len(palette_data):
                    r, g, b = palette_data[i * 3], palette_data[i * 3 + 1], palette_data[i * 3 + 2]
                    palette.colors.append(PaletteColor(r=r, g=g, b=b))
                else:
                    palette.colors.append(PaletteColor())
        
        return model, palette
