"""
VoxelPalette - Color Palette Management
=======================================

Handles color palettes for voxel models, compatible with MagicaVoxel format.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class PaletteColor:
    """Represents a single color in the palette with material properties."""
    r: int = 255
    g: int = 255
    b: int = 255
    a: int = 255
    
    # Material properties (MagicaVoxel compatible)
    material_type: str = "diffuse"  # diffuse, metal, glass, emit
    roughness: float = 1.0
    metallic: float = 0.0
    emission: float = 0.0
    transparency: float = 0.0
    ior: float = 1.5  # Index of refraction for glass
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return color as RGBA tuple."""
        return (self.r, self.g, self.b, self.a)
    
    def to_rgb(self) -> Tuple[int, int, int]:
        """Return color as RGB tuple."""
        return (self.r, self.g, self.b)
    
    def to_hex(self) -> str:
        """Return color as hex string."""
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"
    
    def to_float(self) -> Tuple[float, float, float, float]:
        """Return color as normalized float tuple (0-1 range)."""
        return (self.r / 255.0, self.g / 255.0, self.b / 255.0, self.a / 255.0)
    
    @classmethod
    def from_hex(cls, hex_color: str) -> 'PaletteColor':
        """Create a color from a hex string."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = int(hex_color[6:8], 16) if len(hex_color) >= 8 else 255
        return cls(r=r, g=g, b=b, a=a)
    
    @classmethod
    def from_rgba(cls, r: int, g: int, b: int, a: int = 255) -> 'PaletteColor':
        """Create a color from RGBA values."""
        return cls(r=r, g=g, b=b, a=a)
    
    def blend(self, other: 'PaletteColor', factor: float = 0.5) -> 'PaletteColor':
        """Blend with another color."""
        inv = 1.0 - factor
        return PaletteColor(
            r=int(self.r * inv + other.r * factor),
            g=int(self.g * inv + other.g * factor),
            b=int(self.b * inv + other.b * factor),
            a=int(self.a * inv + other.a * factor)
        )


class VoxelPalette:
    """
    Color palette for voxel models.
    
    Supports 256 colors (index 0 is reserved for empty/transparent).
    Compatible with MagicaVoxel palette format.
    """
    
    def __init__(self):
        """Initialize with default palette."""
        self.colors: List[PaletteColor] = []
        self._init_default_palette()
    
    def _init_default_palette(self):
        """Initialize with a default MagicaVoxel-style palette."""
        # Index 0 is always empty/transparent
        self.colors = [PaletteColor(r=0, g=0, b=0, a=0)]
        
        # Generate a comprehensive default palette
        # Basic colors (1-16)
        basic_colors = [
            (255, 255, 255),  # White
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (128, 128, 128),  # Gray
            (192, 192, 192),  # Light gray
            (128, 0, 0),      # Dark red
            (0, 128, 0),      # Dark green
            (0, 0, 128),      # Dark blue
            (128, 128, 0),    # Olive
            (128, 0, 128),    # Purple
            (0, 128, 128),    # Teal
            (0, 0, 0),        # Black
        ]
        
        for r, g, b in basic_colors:
            self.colors.append(PaletteColor(r=r, g=g, b=b))
        
        # Extended palette with gradients and variations
        # Reds (17-32)
        for i in range(16):
            shade = int(255 * (i + 1) / 16)
            self.colors.append(PaletteColor(r=shade, g=int(shade*0.2), b=int(shade*0.2)))
        
        # Greens (33-48)
        for i in range(16):
            shade = int(255 * (i + 1) / 16)
            self.colors.append(PaletteColor(r=int(shade*0.2), g=shade, b=int(shade*0.2)))
        
        # Blues (49-64)
        for i in range(16):
            shade = int(255 * (i + 1) / 16)
            self.colors.append(PaletteColor(r=int(shade*0.2), g=int(shade*0.2), b=shade))
        
        # Yellows/Oranges (65-80)
        for i in range(16):
            shade = int(255 * (i + 1) / 16)
            self.colors.append(PaletteColor(r=shade, g=int(shade*0.7), b=int(shade*0.1)))
        
        # Purples (81-96)
        for i in range(16):
            shade = int(255 * (i + 1) / 16)
            self.colors.append(PaletteColor(r=int(shade*0.7), g=int(shade*0.2), b=shade))
        
        # Cyans (97-112)
        for i in range(16):
            shade = int(255 * (i + 1) / 16)
            self.colors.append(PaletteColor(r=int(shade*0.2), g=shade, b=shade))
        
        # Grays (113-128)
        for i in range(16):
            shade = int(255 * (i + 1) / 16)
            self.colors.append(PaletteColor(r=shade, g=shade, b=shade))
        
        # Earth tones (129-144)
        earth_base = [(139, 90, 43), (160, 82, 45), (210, 180, 140), (188, 143, 143),
                      (205, 133, 63), (244, 164, 96), (222, 184, 135), (245, 222, 179),
                      (139, 119, 101), (160, 120, 90), (180, 140, 100), (200, 160, 120),
                      (140, 100, 70), (120, 80, 50), (100, 60, 30), (80, 40, 20)]
        for r, g, b in earth_base:
            self.colors.append(PaletteColor(r=r, g=g, b=b))
        
        # Pastels (145-160)
        pastels = [(255, 182, 193), (255, 218, 185), (255, 250, 205), (144, 238, 144),
                   (173, 216, 230), (221, 160, 221), (255, 228, 225), (240, 255, 255),
                   (255, 240, 245), (255, 245, 238), (240, 255, 240), (245, 255, 250),
                   (240, 248, 255), (248, 248, 255), (255, 250, 250), (253, 245, 230)]
        for r, g, b in pastels:
            self.colors.append(PaletteColor(r=r, g=g, b=b))
        
        # Fill remaining slots with rainbow gradient
        while len(self.colors) < 256:
            idx = len(self.colors) - 161
            hue = (idx * 3) % 360
            r, g, b = self._hsv_to_rgb(hue, 0.8, 0.9)
            self.colors.append(PaletteColor(r=int(r*255), g=int(g*255), b=int(b*255)))
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        """Convert HSV to RGB."""
        import colorsys
        return colorsys.hsv_to_rgb(h / 360.0, s, v)
    
    def get_color(self, index: int) -> PaletteColor:
        """Get a color by index."""
        if 0 <= index < len(self.colors):
            return self.colors[index]
        return self.colors[0]
    
    def set_color(self, index: int, r_or_color, g: int = None, b: int = None, a: int = 255):
        """
        Set a color at the specified index.
        
        Can be called with:
            set_color(index, PaletteColor)
            set_color(index, r, g, b)
            set_color(index, r, g, b, a)
        """
        while len(self.colors) <= index:
            self.colors.append(PaletteColor())
        
        if isinstance(r_or_color, PaletteColor):
            self.colors[index] = r_or_color
        else:
            # r_or_color is actually r value
            self.colors[index] = PaletteColor(r=r_or_color, g=g, b=b, a=a)
    
    def load_default(self):
        """Reset to default palette."""
        self.colors = []
        self._init_default_palette()
    
    def sort_by_hue(self):
        """Sort palette colors by hue."""
        import colorsys
        
        def get_hue(color):
            if color.r == color.g == color.b == 0:
                return (0, 0, 0)  # Black at start
            r, g, b = color.r / 255, color.g / 255, color.b / 255
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            return (h, s, v)
        
        # Keep index 0 (transparent) at start
        transparent = self.colors[0] if self.colors else PaletteColor(0, 0, 0, 0)
        remaining = self.colors[1:] if len(self.colors) > 1 else []
        
        # Sort by hue
        remaining.sort(key=get_hue)
        
        self.colors = [transparent] + remaining
    
    def get_rgba_array(self) -> np.ndarray:
        """Get the palette as a numpy array of RGBA values."""
        return np.array([c.to_tuple() for c in self.colors], dtype=np.uint8)
    
    def get_rgb_array(self) -> np.ndarray:
        """Get the palette as a numpy array of RGB values."""
        return np.array([c.to_rgb() for c in self.colors], dtype=np.uint8)
    
    def find_nearest_color(self, r: int, g: int, b: int, 
                           start_index: int = 1) -> int:
        """
        Find the nearest palette color to the given RGB value.
        
        Args:
            r, g, b: Target RGB values
            start_index: Starting index (default 1 to skip transparent)
            
        Returns:
            Index of the nearest color
        """
        min_dist = float('inf')
        nearest_idx = start_index
        
        for i in range(start_index, len(self.colors)):
            c = self.colors[i]
            dist = (c.r - r) ** 2 + (c.g - g) ** 2 + (c.b - b) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def load_from_file(self, filepath: str):
        """Load palette from various file formats."""
        ext = filepath.lower().split('.')[-1]
        
        if ext == 'pal':
            self._load_pal(filepath)
        elif ext == 'png':
            self._load_from_image(filepath)
        elif ext == 'gpl':
            self._load_gimp_palette(filepath)
        elif ext == 'aco':
            self._load_photoshop_palette(filepath)
        else:
            raise ValueError(f"Unsupported palette format: {ext}")
    
    def _load_pal(self, filepath: str):
        """Load a .pal palette file."""
        with open(filepath, 'rb') as f:
            data = f.read()
        
        # Reset palette
        self.colors = [PaletteColor(r=0, g=0, b=0, a=0)]
        
        # Read RGB triplets
        for i in range(0, min(len(data), 768), 3):
            if len(self.colors) >= 256:
                break
            r, g, b = data[i], data[i+1], data[i+2]
            self.colors.append(PaletteColor(r=r, g=g, b=b))
    
    def _load_from_image(self, filepath: str):
        """Load palette from an image file (uses first row or column)."""
        from PIL import Image
        
        img = Image.open(filepath).convert('RGBA')
        pixels = list(img.getdata())
        
        # Reset palette
        self.colors = [PaletteColor(r=0, g=0, b=0, a=0)]
        
        # Extract unique colors (up to 255)
        seen = set()
        for r, g, b, a in pixels:
            if len(self.colors) >= 256:
                break
            if (r, g, b) not in seen:
                seen.add((r, g, b))
                self.colors.append(PaletteColor(r=r, g=g, b=b, a=a))
    
    def _load_gimp_palette(self, filepath: str):
        """Load a GIMP .gpl palette file."""
        self.colors = [PaletteColor(r=0, g=0, b=0, a=0)]
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('GIMP') or line.startswith('Name:') or line.startswith('Columns:'):
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                        self.colors.append(PaletteColor(r=r, g=g, b=b))
                        if len(self.colors) >= 256:
                            break
                    except ValueError:
                        continue
    
    def _load_photoshop_palette(self, filepath: str):
        """Load an Adobe Photoshop .aco palette file."""
        import struct
        
        self.colors = [PaletteColor(r=0, g=0, b=0, a=0)]
        
        with open(filepath, 'rb') as f:
            version, count = struct.unpack('>HH', f.read(4))
            
            for _ in range(count):
                if len(self.colors) >= 256:
                    break
                    
                color_space, w, x, y, z = struct.unpack('>HHHHH', f.read(10))
                
                if color_space == 0:  # RGB
                    r = w >> 8
                    g = x >> 8
                    b = y >> 8
                    self.colors.append(PaletteColor(r=r, g=g, b=b))
    
    def save_to_file(self, filepath: str):
        """Save palette to file."""
        ext = filepath.lower().split('.')[-1]
        
        if ext == 'pal':
            self._save_pal(filepath)
        elif ext == 'png':
            self._save_as_image(filepath)
        elif ext == 'gpl':
            self._save_gimp_palette(filepath)
        else:
            raise ValueError(f"Unsupported palette format: {ext}")
    
    def _save_pal(self, filepath: str):
        """Save as .pal file."""
        data = bytearray()
        for i in range(256):
            if i < len(self.colors):
                c = self.colors[i]
                data.extend([c.r, c.g, c.b])
            else:
                data.extend([0, 0, 0])
        
        with open(filepath, 'wb') as f:
            f.write(data)
    
    def _save_as_image(self, filepath: str):
        """Save palette as a PNG image."""
        from PIL import Image
        
        # Create 16x16 grid
        img = Image.new('RGBA', (16, 16))
        pixels = []
        
        for i in range(256):
            if i < len(self.colors):
                pixels.append(self.colors[i].to_tuple())
            else:
                pixels.append((0, 0, 0, 255))
        
        img.putdata(pixels)
        img.save(filepath)
    
    def _save_gimp_palette(self, filepath: str):
        """Save as GIMP .gpl palette."""
        with open(filepath, 'w') as f:
            f.write("GIMP Palette\n")
            f.write("Name: VoxEdit Palette\n")
            f.write("Columns: 16\n")
            f.write("#\n")
            
            for i, c in enumerate(self.colors[1:], 1):  # Skip index 0
                f.write(f"{c.r:3d} {c.g:3d} {c.b:3d}  Color {i}\n")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize palette to dictionary."""
        return {
            'colors': [(c.r, c.g, c.b, c.a) for c in self.colors],
            'materials': [
                {
                    'type': c.material_type,
                    'roughness': c.roughness,
                    'metallic': c.metallic,
                    'emission': c.emission,
                    'transparency': c.transparency,
                    'ior': c.ior
                }
                for c in self.colors
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoxelPalette':
        """Deserialize palette from dictionary."""
        palette = cls.__new__(cls)
        palette.colors = []
        
        colors = data.get('colors', [])
        materials = data.get('materials', [])
        
        for i, (r, g, b, a) in enumerate(colors):
            color = PaletteColor(r=r, g=g, b=b, a=a)
            if i < len(materials):
                mat = materials[i]
                color.material_type = mat.get('type', 'diffuse')
                color.roughness = mat.get('roughness', 1.0)
                color.metallic = mat.get('metallic', 0.0)
                color.emission = mat.get('emission', 0.0)
                color.transparency = mat.get('transparency', 0.0)
                color.ior = mat.get('ior', 1.5)
            palette.colors.append(color)
        
        return palette
    
    @classmethod
    def create_minecraft_palette(cls) -> 'VoxelPalette':
        """Create a palette with Minecraft block colors."""
        palette = cls.__new__(cls)
        palette.colors = [PaletteColor(r=0, g=0, b=0, a=0)]  # Empty
        
        # Minecraft block colors (approximations)
        minecraft_colors = [
            (125, 125, 125),   # Stone
            (134, 96, 67),     # Dirt
            (118, 179, 76),    # Grass
            (162, 130, 79),    # Sand
            (103, 103, 103),   # Gravel
            (255, 216, 0),     # Gold ore
            (216, 216, 216),   # Iron ore
            (0, 0, 0),         # Coal ore
            (102, 81, 51),     # Oak log
            (60, 192, 41),     # Oak leaves
            (0, 0, 255),       # Lapis lazuli
            (29, 151, 45),     # Emerald
            (150, 67, 22),     # Red sandstone
            (170, 166, 157),   # Andesite
            (188, 152, 98),    # Birch planks
            (255, 255, 255),   # Snow
            (57, 41, 35),      # Soul sand
            (207, 213, 214),   # Quartz
            (119, 86, 59),     # Dark oak
            (208, 127, 93),    # Acacia
            (60, 31, 43),      # Crimson
            (43, 104, 99),     # Warped
            (0, 139, 139),     # Prismarine
            (87, 59, 12),      # Jungle log
            (156, 81, 36),     # Copper
            (47, 47, 47),      # Deepslate
            (106, 76, 54),     # Mud
            (222, 177, 144),   # Calcite
            (42, 42, 42),      # Tuff
            (89, 117, 89),     # Dripstone
            (194, 178, 128),   # Sandstone
            (155, 155, 155),   # Cobblestone
            (195, 195, 195),   # Smooth stone
            (97, 85, 85),      # Brick
            (80, 80, 80),      # Obsidian
            (138, 138, 138),   # Mossy cobblestone
            (204, 76, 76),     # Red wool
            (229, 144, 76),    # Orange wool
            (204, 204, 76),    # Yellow wool
            (76, 178, 76),     # Lime wool
            (76, 204, 204),    # Cyan wool
            (102, 127, 204),   # Light blue wool
            (127, 76, 204),    # Purple wool
            (229, 127, 204),   # Pink wool
            (76, 76, 204),     # Blue wool
            (102, 51, 0),      # Brown wool
            (76, 127, 76),     # Green wool
            (51, 51, 51),      # Black wool
        ]
        
        for r, g, b in minecraft_colors:
            palette.colors.append(PaletteColor(r=r, g=g, b=b))
        
        # Fill rest with variations
        while len(palette.colors) < 256:
            palette.colors.append(PaletteColor(
                r=128 + (len(palette.colors) % 128),
                g=128 + ((len(palette.colors) * 3) % 128),
                b=128 + ((len(palette.colors) * 7) % 128)
            ))
        
        return palette
