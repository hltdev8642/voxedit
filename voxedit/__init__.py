"""
VoxEdit - Professional Teardown VOX Voxel Editor
================================================

A comprehensive 3D voxel editor supporting multiple formats including:
- MagicaVoxel .vox (Teardown compatible)
- Minecraft schematics (.schematic, .nbt, .schem)
- Minecraft world regions (.mca, .mcr)
- Standard 3D formats (OBJ, STL, PLY, GLTF)
- Voxel formats (Binvox, KVX, Qubicle)

Author: VoxEdit Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "VoxEdit Team"
__license__ = "MIT"

from voxedit.core.voxel_model import VoxelModel
from voxedit.core.palette import VoxelPalette

__all__ = ['VoxelModel', 'VoxelPalette', '__version__']
