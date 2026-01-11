"""
Minecraft Format Handlers
=========================

Support for Minecraft schematic and world formats:
- .schematic (MCEdit/WorldEdit classic format)
- .schem (Sponge Schematic format)
- .nbt (NBT structure files)
- .mca/.mcr (Minecraft region/anvil files)

Uses nbtlib for NBT parsing.
"""

import struct
import gzip
import zlib
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

try:
    import nbtlib
    HAS_NBTLIB = True
except ImportError:
    HAS_NBTLIB = False

from voxedit.core.voxel_model import VoxelModel
from voxedit.core.palette import VoxelPalette, PaletteColor


# Minecraft block ID to color mapping (simplified)
MINECRAFT_BLOCK_COLORS = {
    0: (0, 0, 0, 0),           # Air
    1: (125, 125, 125, 255),   # Stone
    2: (118, 179, 76, 255),    # Grass
    3: (134, 96, 67, 255),     # Dirt
    4: (155, 155, 155, 255),   # Cobblestone
    5: (188, 152, 98, 255),    # Oak Planks
    6: (0, 100, 0, 255),       # Sapling
    7: (50, 50, 50, 255),      # Bedrock
    8: (64, 64, 255, 128),     # Water
    9: (64, 64, 255, 128),     # Stationary Water
    10: (255, 128, 0, 255),    # Lava
    11: (255, 128, 0, 255),    # Stationary Lava
    12: (219, 211, 160, 255),  # Sand
    13: (128, 128, 128, 255),  # Gravel
    14: (255, 216, 0, 255),    # Gold Ore
    15: (216, 216, 216, 255),  # Iron Ore
    16: (70, 70, 70, 255),     # Coal Ore
    17: (102, 81, 51, 255),    # Oak Log
    18: (60, 192, 41, 255),    # Oak Leaves
    19: (181, 180, 89, 255),   # Sponge
    20: (200, 220, 255, 80),   # Glass
    21: (0, 0, 160, 255),      # Lapis Ore
    22: (38, 97, 156, 255),    # Lapis Block
    23: (90, 90, 90, 255),     # Dispenser
    24: (194, 178, 128, 255),  # Sandstone
    25: (108, 71, 47, 255),    # Note Block
    26: (155, 35, 53, 255),    # Bed
    35: (221, 221, 221, 255),  # White Wool
    41: (255, 230, 70, 255),   # Gold Block
    42: (220, 220, 220, 255),  # Iron Block
    43: (200, 200, 200, 255),  # Stone Slab
    44: (200, 200, 200, 255),  # Stone Slab
    45: (155, 105, 95, 255),   # Brick
    46: (200, 50, 50, 255),    # TNT
    47: (150, 115, 70, 255),   # Bookshelf
    48: (90, 110, 90, 255),    # Mossy Cobblestone
    49: (30, 20, 40, 255),     # Obsidian
    50: (255, 200, 100, 255),  # Torch
    52: (50, 80, 80, 255),     # Monster Spawner
    54: (145, 110, 45, 255),   # Chest
    56: (100, 210, 240, 255),  # Diamond Ore
    57: (120, 225, 240, 255),  # Diamond Block
    58: (110, 70, 40, 255),    # Crafting Table
    61: (100, 100, 100, 255),  # Furnace
    62: (100, 100, 100, 255),  # Burning Furnace
    73: (145, 55, 55, 255),    # Redstone Ore
    74: (165, 65, 65, 255),    # Lit Redstone Ore
    79: (140, 180, 255, 200),  # Ice
    80: (250, 250, 255, 255),  # Snow Block
    81: (17, 108, 23, 255),    # Cactus
    82: (160, 165, 180, 255),  # Clay
    84: (100, 65, 40, 255),    # Jukebox
    86: (200, 120, 0, 255),    # Pumpkin
    87: (130, 60, 60, 255),    # Netherrack
    88: (85, 65, 55, 255),     # Soul Sand
    89: (220, 180, 50, 255),   # Glowstone
    91: (230, 145, 30, 255),   # Jack o'Lantern
    98: (90, 90, 90, 255),     # Stone Bricks
    110: (65, 90, 65, 255),    # Mycelium
    112: (55, 30, 35, 255),    # Nether Brick
    121: (225, 225, 200, 255), # End Stone
    129: (0, 155, 90, 255),    # Emerald Ore
    133: (75, 220, 115, 255),  # Emerald Block
    152: (175, 35, 25, 255),   # Redstone Block
    153: (200, 160, 140, 255), # Nether Quartz Ore
    155: (235, 230, 220, 255), # Quartz Block
    159: (200, 180, 165, 255), # Terracotta
    162: (60, 45, 30, 255),    # Dark Oak Log
    172: (150, 85, 60, 255),   # Hardened Clay
    173: (20, 20, 20, 255),    # Coal Block
    174: (130, 180, 255, 200), # Packed Ice
    179: (185, 90, 50, 255),   # Red Sandstone
    201: (190, 105, 190, 255), # Purpur Block
    206: (210, 195, 130, 255), # End Stone Bricks
    213: (200, 70, 40, 255),   # Magma Block
    214: (57, 41, 35, 255),    # Nether Wart Block
    215: (110, 30, 30, 255),   # Red Nether Brick
    236: (255, 255, 255, 255), # White Concrete
    251: (255, 255, 255, 255), # White Concrete (pre-1.12)
}


@dataclass
class MinecraftBlock:
    """Represents a Minecraft block."""
    x: int
    y: int
    z: int
    block_id: int
    data: int = 0
    nbt: Optional[Dict] = None


class MinecraftSchematic:
    """
    Handler for Minecraft schematic files.
    
    Supports:
    - Classic .schematic format (MCEdit/WorldEdit)
    - Sponge .schem format (modern)
    - Structure .nbt files
    """
    
    @classmethod
    def load(cls, filepath: str) -> Tuple[VoxelModel, VoxelPalette]:
        """
        Load a Minecraft schematic file.
        
        Args:
            filepath: Path to the schematic file
            
        Returns:
            Tuple of (VoxelModel, VoxelPalette)
        """
        ext = Path(filepath).suffix.lower()
        
        if ext == '.schematic':
            return cls._load_classic_schematic(filepath)
        elif ext == '.schem':
            return cls._load_sponge_schematic(filepath)
        elif ext == '.nbt':
            return cls._load_structure_nbt(filepath)
        else:
            raise ValueError(f"Unsupported schematic format: {ext}")
    
    @classmethod
    def _load_classic_schematic(cls, filepath: str) -> Tuple[VoxelModel, VoxelPalette]:
        """Load classic MCEdit .schematic format."""
        if not HAS_NBTLIB:
            raise ImportError("nbtlib is required for schematic loading. Install with: pip install nbtlib")
        
        # Load NBT file
        nbt_file = nbtlib.load(filepath)
        schematic = nbt_file['Schematic'] if 'Schematic' in nbt_file else nbt_file
        
        width = int(schematic['Width'])
        height = int(schematic['Height'])
        length = int(schematic['Length'])
        
        blocks = np.array(schematic['Blocks'], dtype=np.int16)
        if 'AddBlocks' in schematic:
            add_blocks = np.array(schematic['AddBlocks'], dtype=np.int16)
            # Expand add blocks to full array
            expanded = np.zeros(len(blocks), dtype=np.int16)
            for i in range(len(add_blocks)):
                expanded[i * 2] = (add_blocks[i] >> 4) & 0xF
                expanded[i * 2 + 1] = add_blocks[i] & 0xF
            blocks = blocks + (expanded[:len(blocks)] << 8)
        
        data = np.array(schematic.get('Data', np.zeros_like(blocks)), dtype=np.uint8)
        
        # Create model
        model = VoxelModel.create(width, height, length, name=Path(filepath).stem)
        
        # Create palette from block colors
        palette = VoxelPalette()
        palette.colors = [PaletteColor(r=0, g=0, b=0, a=0)]  # Empty
        
        block_to_palette = {0: 0}  # Air is always 0
        next_palette_idx = 1
        
        # Process blocks
        for i, (block_id, block_data) in enumerate(zip(blocks, data)):
            if block_id == 0:  # Skip air
                continue
            
            # Calculate position (YZX order in schematic)
            y = i // (width * length)
            remainder = i % (width * length)
            z = remainder // width
            x = remainder % width
            
            # Get or create palette entry
            block_key = int(block_id)
            if block_key not in block_to_palette:
                if next_palette_idx < 256:
                    color = MINECRAFT_BLOCK_COLORS.get(block_key, (128, 128, 128, 255))
                    palette.colors.append(PaletteColor(r=color[0], g=color[1], b=color[2], a=color[3]))
                    block_to_palette[block_key] = next_palette_idx
                    next_palette_idx += 1
                else:
                    # Find closest existing color
                    color = MINECRAFT_BLOCK_COLORS.get(block_key, (128, 128, 128, 255))
                    block_to_palette[block_key] = palette.find_nearest_color(color[0], color[1], color[2])
            
            model.set_voxel(x, y, z, block_to_palette[block_key])
        
        # Fill remaining palette slots
        while len(palette.colors) < 256:
            palette.colors.append(PaletteColor())
        
        model.commit()
        return model, palette
    
    @classmethod
    def _load_sponge_schematic(cls, filepath: str) -> Tuple[VoxelModel, VoxelPalette]:
        """Load Sponge .schem format."""
        if not HAS_NBTLIB:
            raise ImportError("nbtlib is required for schematic loading. Install with: pip install nbtlib")
        
        nbt_file = nbtlib.load(filepath)
        schematic = nbt_file['Schematic'] if 'Schematic' in nbt_file else nbt_file
        
        width = int(schematic['Width'])
        height = int(schematic['Height'])
        length = int(schematic['Length'])
        
        # Get palette
        block_palette = schematic.get('Palette', {})
        palette_map = {int(v): str(k) for k, v in block_palette.items()}
        
        # Get block data
        block_data = np.array(schematic['BlockData'], dtype=np.int32)
        
        # Create model
        model = VoxelModel.create(width, height, length, name=Path(filepath).stem)
        
        # Create color palette
        palette = VoxelPalette()
        palette.colors = [PaletteColor(r=0, g=0, b=0, a=0)]  # Empty
        
        block_name_to_palette = {'minecraft:air': 0}
        next_palette_idx = 1
        
        # Parse block names to get colors
        for idx, block_name in palette_map.items():
            if idx == 0 or 'air' in block_name:
                continue
            
            # Try to map block name to color
            color = cls._get_color_for_block_name(block_name)
            if next_palette_idx < 256:
                palette.colors.append(PaletteColor(r=color[0], g=color[1], b=color[2], a=color[3]))
                block_name_to_palette[block_name] = next_palette_idx
                next_palette_idx += 1
        
        # Process blocks
        for i, palette_idx in enumerate(block_data):
            if palette_idx == 0:
                continue
            
            block_name = palette_map.get(int(palette_idx), 'minecraft:air')
            if 'air' in block_name:
                continue
            
            # Calculate position
            y = i // (width * length)
            remainder = i % (width * length)
            z = remainder // width
            x = remainder % width
            
            voxel_palette_idx = block_name_to_palette.get(block_name, 1)
            model.set_voxel(x, y, z, voxel_palette_idx)
        
        while len(palette.colors) < 256:
            palette.colors.append(PaletteColor())
        
        model.commit()
        return model, palette
    
    @classmethod
    def _load_structure_nbt(cls, filepath: str) -> Tuple[VoxelModel, VoxelPalette]:
        """Load Minecraft structure .nbt format."""
        if not HAS_NBTLIB:
            raise ImportError("nbtlib is required for structure loading. Install with: pip install nbtlib")
        
        nbt_file = nbtlib.load(filepath)
        
        size = [int(x) for x in nbt_file['size']]
        palette_data = nbt_file.get('palette', [])
        blocks_data = nbt_file.get('blocks', [])
        
        model = VoxelModel.create(size[0], size[1], size[2], name=Path(filepath).stem)
        
        palette = VoxelPalette()
        palette.colors = [PaletteColor(r=0, g=0, b=0, a=0)]
        
        # Build palette from block types
        block_type_to_palette = {0: 0}
        next_idx = 1
        
        for i, block_entry in enumerate(palette_data):
            block_name = str(block_entry.get('Name', 'minecraft:air'))
            if 'air' not in block_name and next_idx < 256:
                color = cls._get_color_for_block_name(block_name)
                palette.colors.append(PaletteColor(r=color[0], g=color[1], b=color[2], a=color[3]))
                block_type_to_palette[i] = next_idx
                next_idx += 1
        
        # Place blocks
        for block in blocks_data:
            pos = [int(x) for x in block['pos']]
            state = int(block.get('state', 0))
            
            palette_idx = block_type_to_palette.get(state, 0)
            if palette_idx > 0:
                model.set_voxel(pos[0], pos[1], pos[2], palette_idx)
        
        while len(palette.colors) < 256:
            palette.colors.append(PaletteColor())
        
        model.commit()
        return model, palette
    
    @classmethod
    def _get_color_for_block_name(cls, block_name: str) -> Tuple[int, int, int, int]:
        """Get a color for a Minecraft block name."""
        name = block_name.lower().replace('minecraft:', '')
        
        # Color mapping for common blocks
        color_map = {
            'stone': (125, 125, 125, 255),
            'granite': (155, 100, 80, 255),
            'diorite': (180, 180, 180, 255),
            'andesite': (130, 130, 130, 255),
            'grass_block': (118, 179, 76, 255),
            'dirt': (134, 96, 67, 255),
            'cobblestone': (155, 155, 155, 255),
            'oak_planks': (188, 152, 98, 255),
            'spruce_planks': (115, 85, 50, 255),
            'birch_planks': (230, 215, 160, 255),
            'jungle_planks': (180, 135, 95, 255),
            'acacia_planks': (175, 95, 55, 255),
            'dark_oak_planks': (65, 45, 25, 255),
            'sand': (219, 211, 160, 255),
            'gravel': (128, 128, 128, 255),
            'gold_ore': (255, 216, 0, 255),
            'iron_ore': (216, 216, 216, 255),
            'coal_ore': (70, 70, 70, 255),
            'oak_log': (102, 81, 51, 255),
            'oak_leaves': (60, 192, 41, 255),
            'glass': (200, 220, 255, 80),
            'lapis_block': (38, 97, 156, 255),
            'sandstone': (194, 178, 128, 255),
            'white_wool': (221, 221, 221, 255),
            'orange_wool': (235, 136, 68, 255),
            'magenta_wool': (190, 75, 170, 255),
            'light_blue_wool': (110, 175, 220, 255),
            'yellow_wool': (195, 185, 45, 255),
            'lime_wool': (112, 185, 30, 255),
            'pink_wool': (240, 170, 190, 255),
            'gray_wool': (65, 65, 65, 255),
            'light_gray_wool': (155, 155, 155, 255),
            'cyan_wool': (50, 135, 150, 255),
            'purple_wool': (130, 55, 180, 255),
            'blue_wool': (50, 60, 160, 255),
            'brown_wool': (115, 70, 40, 255),
            'green_wool': (85, 110, 40, 255),
            'red_wool': (155, 45, 45, 255),
            'black_wool': (25, 25, 25, 255),
            'gold_block': (255, 230, 70, 255),
            'iron_block': (220, 220, 220, 255),
            'brick': (155, 105, 95, 255),
            'tnt': (200, 50, 50, 255),
            'mossy_cobblestone': (90, 110, 90, 255),
            'obsidian': (30, 20, 40, 255),
            'diamond_ore': (100, 210, 240, 255),
            'diamond_block': (120, 225, 240, 255),
            'redstone_block': (175, 35, 25, 255),
            'emerald_block': (75, 220, 115, 255),
            'netherrack': (130, 60, 60, 255),
            'soul_sand': (85, 65, 55, 255),
            'glowstone': (220, 180, 50, 255),
            'nether_bricks': (55, 30, 35, 255),
            'end_stone': (225, 225, 200, 255),
            'purpur_block': (190, 105, 190, 255),
            'prismarine': (100, 170, 160, 255),
            'sea_lantern': (180, 220, 210, 255),
            'terracotta': (160, 80, 55, 255),
            'concrete': (200, 200, 200, 255),
            'quartz_block': (235, 230, 220, 255),
        }
        
        # Find matching color
        for key, color in color_map.items():
            if key in name:
                return color
        
        # Default color
        return (128, 128, 128, 255)
    
    @classmethod
    def save(cls, filepath: str, model: VoxelModel, palette: VoxelPalette):
        """
        Save a VoxelModel to a schematic file.
        
        Args:
            filepath: Output file path
            model: VoxelModel to save
            palette: Color palette
        """
        ext = Path(filepath).suffix.lower()
        
        if ext == '.schematic':
            cls._save_classic_schematic(filepath, model, palette)
        elif ext == '.schem':
            cls._save_sponge_schematic(filepath, model, palette)
        elif ext == '.nbt':
            cls._save_structure_nbt(filepath, model, palette)
        else:
            raise ValueError(f"Unsupported schematic format: {ext}")
    
    @classmethod
    def _save_classic_schematic(cls, filepath: str, model: VoxelModel, palette: VoxelPalette):
        """Save to classic MCEdit .schematic format."""
        if not HAS_NBTLIB:
            raise ImportError("nbtlib is required for schematic saving. Install with: pip install nbtlib")
        
        width, height, length = model.size
        
        # Create block and data arrays
        blocks = np.zeros(width * height * length, dtype=np.uint8)
        data = np.zeros(width * height * length, dtype=np.uint8)
        
        # Map palette colors to Minecraft blocks
        palette_to_block = cls._map_palette_to_blocks(palette)
        
        # Fill arrays
        for x, y, z, color_idx in model.get_all_voxels():
            i = y * (width * length) + z * width + x
            block_id = palette_to_block.get(color_idx, 1)  # Default to stone
            blocks[i] = block_id & 0xFF
        
        # Create NBT structure
        schematic = nbtlib.Compound({
            'Schematic': nbtlib.Compound({
                'Width': nbtlib.Short(width),
                'Height': nbtlib.Short(height),
                'Length': nbtlib.Short(length),
                'Materials': nbtlib.String('Alpha'),
                'Blocks': nbtlib.ByteArray(blocks.tolist()),
                'Data': nbtlib.ByteArray(data.tolist()),
                'Entities': nbtlib.List[nbtlib.Compound](),
                'TileEntities': nbtlib.List[nbtlib.Compound](),
            })
        })
        
        schematic.save(filepath)
    
    @classmethod
    def _save_sponge_schematic(cls, filepath: str, model: VoxelModel, palette: VoxelPalette):
        """Save to Sponge .schem format."""
        if not HAS_NBTLIB:
            raise ImportError("nbtlib is required for schematic saving. Install with: pip install nbtlib")
        
        width, height, length = model.size
        
        # Create block palette
        block_palette = {'minecraft:air': nbtlib.Int(0)}
        palette_to_block = cls._map_palette_to_blocks(palette)
        
        # Create block data
        block_data = []
        palette_idx_map = {0: 0}  # Air
        next_idx = 1
        
        for y in range(height):
            for z in range(length):
                for x in range(width):
                    color_idx = model.get_voxel(x, y, z)
                    if color_idx == 0:
                        block_data.append(0)
                    else:
                        block_id = palette_to_block.get(color_idx, 1)
                        block_name = cls._block_id_to_name(block_id)
                        
                        if block_name not in block_palette:
                            block_palette[block_name] = nbtlib.Int(next_idx)
                            palette_idx_map[color_idx] = next_idx
                            next_idx += 1
                        
                        block_data.append(int(block_palette[block_name]))
        
        schematic = nbtlib.Compound({
            'Schematic': nbtlib.Compound({
                'Version': nbtlib.Int(2),
                'DataVersion': nbtlib.Int(2586),
                'Width': nbtlib.Short(width),
                'Height': nbtlib.Short(height),
                'Length': nbtlib.Short(length),
                'Palette': nbtlib.Compound(block_palette),
                'BlockData': nbtlib.ByteArray(block_data),
            })
        })
        
        schematic.save(filepath)
    
    @classmethod
    def _save_structure_nbt(cls, filepath: str, model: VoxelModel, palette: VoxelPalette):
        """Save to Minecraft structure .nbt format."""
        if not HAS_NBTLIB:
            raise ImportError("nbtlib is required for structure saving. Install with: pip install nbtlib")
        
        # Create palette
        block_palette = []
        palette_to_block = cls._map_palette_to_blocks(palette)
        palette_idx_to_state = {}
        
        # Add air
        block_palette.append(nbtlib.Compound({'Name': nbtlib.String('minecraft:air')}))
        palette_idx_to_state[0] = 0
        next_state = 1
        
        # Create blocks list
        blocks = []
        
        for x, y, z, color_idx in model.get_all_voxels():
            block_id = palette_to_block.get(color_idx, 1)
            block_name = cls._block_id_to_name(block_id)
            
            if color_idx not in palette_idx_to_state:
                block_palette.append(nbtlib.Compound({'Name': nbtlib.String(block_name)}))
                palette_idx_to_state[color_idx] = next_state
                next_state += 1
            
            blocks.append(nbtlib.Compound({
                'pos': nbtlib.List[nbtlib.Int]([nbtlib.Int(x), nbtlib.Int(y), nbtlib.Int(z)]),
                'state': nbtlib.Int(palette_idx_to_state[color_idx])
            }))
        
        structure = nbtlib.Compound({
            'size': nbtlib.List[nbtlib.Int]([
                nbtlib.Int(model.size[0]),
                nbtlib.Int(model.size[1]),
                nbtlib.Int(model.size[2])
            ]),
            'palette': nbtlib.List[nbtlib.Compound](block_palette),
            'blocks': nbtlib.List[nbtlib.Compound](blocks),
            'entities': nbtlib.List[nbtlib.Compound](),
            'DataVersion': nbtlib.Int(2586)
        })
        
        nbtlib.File(structure).save(filepath)
    
    @classmethod
    def _map_palette_to_blocks(cls, palette: VoxelPalette) -> Dict[int, int]:
        """Map palette indices to Minecraft block IDs based on color."""
        result = {0: 0}  # Empty/Air
        
        for i in range(1, len(palette.colors)):
            color = palette.colors[i]
            best_match = 1  # Default to stone
            best_dist = float('inf')
            
            for block_id, (r, g, b, a) in MINECRAFT_BLOCK_COLORS.items():
                if block_id == 0:
                    continue
                dist = (color.r - r)**2 + (color.g - g)**2 + (color.b - b)**2
                if dist < best_dist:
                    best_dist = dist
                    best_match = block_id
            
            result[i] = best_match
        
        return result
    
    @classmethod
    def _block_id_to_name(cls, block_id: int) -> str:
        """Convert block ID to Minecraft block name."""
        block_names = {
            0: 'minecraft:air',
            1: 'minecraft:stone',
            2: 'minecraft:grass_block',
            3: 'minecraft:dirt',
            4: 'minecraft:cobblestone',
            5: 'minecraft:oak_planks',
            7: 'minecraft:bedrock',
            12: 'minecraft:sand',
            13: 'minecraft:gravel',
            14: 'minecraft:gold_ore',
            15: 'minecraft:iron_ore',
            16: 'minecraft:coal_ore',
            17: 'minecraft:oak_log',
            18: 'minecraft:oak_leaves',
            20: 'minecraft:glass',
            22: 'minecraft:lapis_block',
            24: 'minecraft:sandstone',
            35: 'minecraft:white_wool',
            41: 'minecraft:gold_block',
            42: 'minecraft:iron_block',
            45: 'minecraft:bricks',
            48: 'minecraft:mossy_cobblestone',
            49: 'minecraft:obsidian',
            56: 'minecraft:diamond_ore',
            57: 'minecraft:diamond_block',
            79: 'minecraft:ice',
            80: 'minecraft:snow_block',
            82: 'minecraft:clay',
            87: 'minecraft:netherrack',
            89: 'minecraft:glowstone',
            98: 'minecraft:stone_bricks',
            112: 'minecraft:nether_bricks',
            121: 'minecraft:end_stone',
            133: 'minecraft:emerald_block',
            152: 'minecraft:redstone_block',
            155: 'minecraft:quartz_block',
        }
        
        return block_names.get(block_id, 'minecraft:stone')


class MinecraftRegion:
    """
    Handler for Minecraft region files (.mca, .mcr).
    
    Supports reading Anvil format region files containing
    multiple chunks of world data.
    """
    
    SECTOR_SIZE = 4096
    CHUNK_WIDTH = 16
    CHUNK_HEIGHT = 256
    
    @classmethod
    def load(cls, filepath: str, 
             chunk_range: Optional[Tuple[int, int, int, int]] = None) -> Tuple[VoxelModel, VoxelPalette]:
        """
        Load a Minecraft region file.
        
        Args:
            filepath: Path to the region file
            chunk_range: Optional (x1, z1, x2, z2) range of chunks to load
            
        Returns:
            Tuple of (VoxelModel, VoxelPalette)
        """
        if not HAS_NBTLIB:
            raise ImportError("nbtlib is required for region loading. Install with: pip install nbtlib")
        
        # Read region header
        with open(filepath, 'rb') as f:
            # Read location table (1024 entries of 4 bytes each)
            locations = []
            for _ in range(1024):
                loc = struct.unpack('>I', f.read(4))[0]
                offset = (loc >> 8) & 0xFFFFFF
                size = loc & 0xFF
                locations.append((offset, size))
            
            # Read timestamps (skip)
            f.read(4096)
            
            # Find non-empty chunks
            chunks = []
            for i, (offset, size) in enumerate(locations):
                if offset == 0 or size == 0:
                    continue
                
                chunk_x = i % 32
                chunk_z = i // 32
                
                # Check if in range
                if chunk_range is not None:
                    x1, z1, x2, z2 = chunk_range
                    if not (x1 <= chunk_x <= x2 and z1 <= chunk_z <= z2):
                        continue
                
                # Read chunk data
                f.seek(offset * cls.SECTOR_SIZE)
                length = struct.unpack('>I', f.read(4))[0]
                compression = struct.unpack('B', f.read(1))[0]
                
                compressed_data = f.read(length - 1)
                
                if compression == 2:  # Zlib
                    chunk_data = zlib.decompress(compressed_data)
                elif compression == 1:  # Gzip
                    chunk_data = gzip.decompress(compressed_data)
                else:
                    continue
                
                # Parse NBT
                try:
                    chunk_nbt = nbtlib.File.parse(nbtlib.BytesIO(chunk_data))
                    chunks.append((chunk_x, chunk_z, chunk_nbt))
                except Exception as e:
                    print(f"Warning: Failed to parse chunk at ({chunk_x}, {chunk_z}): {e}")
        
        if not chunks:
            raise ValueError("No valid chunks found in region file")
        
        # Calculate world bounds
        min_cx = min(c[0] for c in chunks)
        max_cx = max(c[0] for c in chunks)
        min_cz = min(c[1] for c in chunks)
        max_cz = max(c[1] for c in chunks)
        
        world_width = (max_cx - min_cx + 1) * cls.CHUNK_WIDTH
        world_length = (max_cz - min_cz + 1) * cls.CHUNK_WIDTH
        
        model = VoxelModel.create(world_width, cls.CHUNK_HEIGHT, world_length,
                                   name=Path(filepath).stem)
        
        palette = VoxelPalette.create_minecraft_palette()
        
        # Process chunks
        for chunk_x, chunk_z, chunk_nbt in chunks:
            cls._process_chunk(model, chunk_nbt, 
                              (chunk_x - min_cx) * cls.CHUNK_WIDTH,
                              (chunk_z - min_cz) * cls.CHUNK_WIDTH,
                              palette)
        
        model.commit()
        return model, palette
    
    @classmethod
    def _process_chunk(cls, model: VoxelModel, chunk_nbt, 
                       offset_x: int, offset_z: int, palette: VoxelPalette):
        """Process a single chunk and add to model."""
        try:
            level = chunk_nbt.get('Level', chunk_nbt)
            sections = level.get('Sections', [])
            
            for section in sections:
                y_offset = int(section.get('Y', 0)) * 16
                if y_offset < 0 or y_offset >= model.size[1]:
                    continue
                
                # Try to read block data
                if 'Palette' in section and 'BlockStates' in section:
                    # New format (1.13+)
                    cls._process_section_new(model, section, offset_x, y_offset, offset_z, palette)
                elif 'Blocks' in section:
                    # Old format (pre-1.13)
                    cls._process_section_old(model, section, offset_x, y_offset, offset_z, palette)
        except Exception as e:
            print(f"Warning: Failed to process chunk: {e}")
    
    @classmethod
    def _process_section_old(cls, model: VoxelModel, section, 
                             offset_x: int, y_offset: int, offset_z: int,
                             palette: VoxelPalette):
        """Process old format chunk section (pre-1.13)."""
        blocks = np.array(section['Blocks'], dtype=np.uint8)
        
        for i, block_id in enumerate(blocks):
            if block_id == 0:  # Air
                continue
            
            y = i // 256
            remainder = i % 256
            z = remainder // 16
            x = remainder % 16
            
            world_x = offset_x + x
            world_y = y_offset + y
            world_z = offset_z + z
            
            if model.is_valid_position(world_x, world_y, world_z):
                color = MINECRAFT_BLOCK_COLORS.get(block_id, (128, 128, 128, 255))
                color_idx = palette.find_nearest_color(color[0], color[1], color[2])
                model.set_voxel(world_x, world_y, world_z, color_idx)
    
    @classmethod
    def _process_section_new(cls, model: VoxelModel, section,
                             offset_x: int, y_offset: int, offset_z: int,
                             palette: VoxelPalette):
        """Process new format chunk section (1.13+)."""
        section_palette = section['Palette']
        block_states = np.array(section['BlockStates'], dtype=np.int64)
        
        # Calculate bits per block
        palette_size = len(section_palette)
        bits = max(4, (palette_size - 1).bit_length())
        
        # Unpack block states
        blocks_per_long = 64 // bits
        mask = (1 << bits) - 1
        
        for i in range(4096):  # 16x16x16
            long_index = i // blocks_per_long
            bit_offset = (i % blocks_per_long) * bits
            
            if long_index >= len(block_states):
                break
            
            palette_idx = (block_states[long_index] >> bit_offset) & mask
            
            if palette_idx >= len(section_palette):
                continue
            
            block_entry = section_palette[int(palette_idx)]
            block_name = str(block_entry.get('Name', 'minecraft:air'))
            
            if 'air' in block_name:
                continue
            
            y = i // 256
            remainder = i % 256
            z = remainder // 16
            x = remainder % 16
            
            world_x = offset_x + x
            world_y = y_offset + y
            world_z = offset_z + z
            
            if model.is_valid_position(world_x, world_y, world_z):
                color = MinecraftSchematic._get_color_for_block_name(block_name)
                color_idx = palette.find_nearest_color(color[0], color[1], color[2])
                model.set_voxel(world_x, world_y, world_z, color_idx)
