"""
VoxEdit Formats Module
======================

File format readers and writers for various voxel and 3D formats.
"""

from pathlib import Path
from typing import Tuple, Optional

from voxedit.formats.vox import VoxFormat
from voxedit.formats.minecraft import MinecraftSchematic, MinecraftRegion
from voxedit.formats.mesh import ObjExporter, StlExporter, PlyExporter, GltfExporter
from voxedit.formats.voxel import BinvoxFormat, KvxFormat, QubicleFormat


class FormatManager:
    """
    Centralized file format manager.
    
    Handles importing and exporting voxel models in various formats.
    """
    
    # Supported import formats
    IMPORT_FORMATS = {
        '.vox': ('MagicaVoxel', VoxFormat),
        '.schematic': ('Minecraft Schematic', MinecraftSchematic),
        '.schem': ('Minecraft Schem', MinecraftSchematic),
        '.litematic': ('Litematica', MinecraftSchematic),
        '.binvox': ('Binvox', BinvoxFormat),
        '.qb': ('Qubicle Binary', QubicleFormat),
        '.kvx': ('KVX Voxel', KvxFormat),
    }
    
    # Supported export formats
    EXPORT_FORMATS = {
        '.vox': ('MagicaVoxel', VoxFormat),
        '.obj': ('Wavefront OBJ', ObjExporter),
        '.stl': ('STL Mesh', StlExporter),
        '.ply': ('PLY Mesh', PlyExporter),
        '.gltf': ('GLTF', GltfExporter),
        '.schematic': ('Minecraft Schematic', MinecraftSchematic),
        '.binvox': ('Binvox', BinvoxFormat),
        '.qb': ('Qubicle Binary', QubicleFormat),
    }
    
    def __init__(self):
        pass
    
    def get_import_filter_string(self) -> str:
        """Get a Qt file dialog filter string for import formats."""
        filters = []
        
        # All supported formats
        all_exts = ' '.join(f'*{ext}' for ext in self.IMPORT_FORMATS.keys())
        filters.append(f"All Voxel Files ({all_exts})")
        
        # Individual formats
        for ext, (name, _) in self.IMPORT_FORMATS.items():
            filters.append(f"{name} (*{ext})")
        
        filters.append("All Files (*)")
        
        return ';;'.join(filters)
    
    def get_export_filter_string(self) -> str:
        """Get a Qt file dialog filter string for export formats."""
        filters = []
        
        for ext, (name, _) in self.EXPORT_FORMATS.items():
            filters.append(f"{name} (*{ext})")
        
        filters.append("All Files (*)")
        
        return ';;'.join(filters)
    
    def import_file(self, filepath: str) -> Tuple['VoxelModel', Optional['VoxelPalette']]:
        """
        Import a voxel model from a file.
        
        Args:
            filepath: Path to the file to import
            
        Returns:
            Tuple of (VoxelModel, VoxelPalette or None)
            
        Raises:
            ValueError: If the file format is not supported
            IOError: If the file cannot be read
        """
        from voxedit.core import VoxelModel, VoxelPalette
        
        path = Path(filepath)
        ext = path.suffix.lower()
        
        if ext not in self.IMPORT_FORMATS:
            raise ValueError(f"Unsupported import format: {ext}")
        
        _, handler_class = self.IMPORT_FORMATS[ext]
        
        # Create handler and read file
        handler = handler_class()
        
        if hasattr(handler, 'read'):
            return handler.read(filepath)
        elif hasattr(handler, 'import_file'):
            return handler.import_file(filepath)
        else:
            raise ValueError(f"Handler for {ext} does not support import")
    
    def export_file(self, filepath: str, model: 'VoxelModel', 
                    palette: Optional['VoxelPalette'] = None):
        """
        Export a voxel model to a file.
        
        Args:
            filepath: Path for the output file
            model: VoxelModel to export
            palette: Optional palette (required for some formats)
            
        Raises:
            ValueError: If the file format is not supported
            IOError: If the file cannot be written
        """
        path = Path(filepath)
        ext = path.suffix.lower()
        
        if ext not in self.EXPORT_FORMATS:
            raise ValueError(f"Unsupported export format: {ext}")
        
        _, handler_class = self.EXPORT_FORMATS[ext]
        
        # Create handler and write file
        handler = handler_class()
        
        if hasattr(handler, 'write'):
            handler.write(filepath, model, palette)
        elif hasattr(handler, 'export_file'):
            handler.export_file(filepath, model, palette)
        elif hasattr(handler, 'export'):
            handler.export(filepath, model, palette)
        else:
            raise ValueError(f"Handler for {ext} does not support export")
    
    def can_import(self, filepath: str) -> bool:
        """Check if a file can be imported."""
        ext = Path(filepath).suffix.lower()
        return ext in self.IMPORT_FORMATS
    
    def can_export(self, filepath: str) -> bool:
        """Check if a file can be exported to the given format."""
        ext = Path(filepath).suffix.lower()
        return ext in self.EXPORT_FORMATS


__all__ = [
    'FormatManager',
    'VoxFormat',
    'MinecraftSchematic', 
    'MinecraftRegion',
    'ObjExporter', 
    'StlExporter', 
    'PlyExporter', 
    'GltfExporter',
    'BinvoxFormat', 
    'KvxFormat', 
    'QubicleFormat'
]
