# VoxEdit - Teardown Voxel Editor

A professional 3D voxel editor with comprehensive format support, designed for creating and editing voxel models compatible with Teardown and other voxel-based applications.

![VoxEdit Screenshot](docs/screenshot.png)

## Features

### 🎨 3D Voxel Editing
- **Full 3D viewport** with OpenGL rendering
- **Orbital camera** controls (rotate, pan, zoom)
- **Multiple editing tools**: Pencil, Line, Box, Sphere, Fill, Eraser, Eyedropper, Paint
- **Brush sizes** from 1-20 voxels
- **Mirror modes** for symmetrical editing
- **Undo/Redo** support with 50-level history

### 📁 Format Support

**Import:**
- MagicaVoxel (`.vox`) - Full support including palettes and materials
- Minecraft Schematic (`.schematic`, `.schem`, `.litematic`)
- Binvox (`.binvox`)
- Qubicle Binary (`.qb`)
- KVX (`.kvx`)

**Export:**
- MagicaVoxel (`.vox`)
- Wavefront OBJ (`.obj`)
- STL (`.stl`)
- PLY (`.ply`)
- GLTF (`.gltf`)
- Minecraft Schematic (`.schematic`)

### 🛠️ Advanced Operations
- **Rotate** (90° increments on any axis)
- **Mirror/Flip** along any axis
- **Hollow/Shell** with adjustable thickness
- **Smooth** using cellular automata
- **Erode/Dilate** morphological operations
- **Color replacement** across model
- **Model resizing** with content preservation

### 🎨 Palette Management
- **256-color palette** (MagicaVoxel compatible)
- **Visual palette grid** with color preview
- **Color picker** with RGB controls
- **Import/Export palettes** (.pal, .png, .gpl)
- **Default palettes** including Minecraft colors

## Installation

### Requirements
- Python 3.10+
- PyQt6
- PyOpenGL
- NumPy
- Pillow
- SciPy (for advanced operations)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/voxedit.git
cd voxedit

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Quick Install

```bash
pip install -r requirements.txt
python main.py
```

## Usage

### Starting the Editor

```bash
# Start with empty 32x32x32 model
python main.py

# Open an existing file
python main.py path/to/model.vox

# Start with custom model size
python main.py --size 64x64x64
```

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New model |
| `Ctrl+O` | Open file |
| `Ctrl+S` | Save |
| `Ctrl+Shift+S` | Save as |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Home` | Reset camera view |
| `G` | Toggle grid |
| `A` | Toggle axes |
| `W` | Toggle wireframe |

### Mouse Controls

| Action | Control |
|--------|---------|
| Orbit camera | Right-click + drag |
| Pan camera | Middle-click + drag |
| Zoom | Mouse wheel |
| Place voxel | Left-click |
| Remove voxel | Left-click with Eraser tool |
| Pick color | Left-click with Eyedropper |

### Tools

- **Select (V)**: Select regions for operations
- **Pencil (P)**: Place individual voxels
- **Line (L)**: Draw lines between points
- **Box (B)**: Draw filled or hollow boxes
- **Sphere (S)**: Draw filled or hollow spheres
- **Fill (F)**: Flood fill connected regions
- **Eraser (E)**: Remove voxels
- **Eyedropper (I)**: Pick color from model
- **Paint (G)**: Paint over existing voxels

## File Formats

### MagicaVoxel (.vox)
The primary format, fully compatible with MagicaVoxel and Teardown.

```python
from voxedit.formats import VoxFormat
from voxedit.core import VoxelModel, VoxelPalette

# Read
handler = VoxFormat()
model, palette = handler.read("model.vox")

# Write
handler.write("output.vox", model, palette)
```

### Minecraft Schematics
Import Minecraft builds and structures.

```python
from voxedit.formats import MinecraftSchematic

handler = MinecraftSchematic()
model, palette = handler.read("build.schematic")
```

### Mesh Export
Export for 3D printing or rendering.

```python
from voxedit.formats import ObjExporter, StlExporter

# Export as OBJ
obj = ObjExporter()
obj.export("model.obj", model, palette)

# Export as STL (for 3D printing)
stl = StlExporter()
stl.export("model.stl", model, palette)
```

## API Reference

### VoxelModel

```python
from voxedit.core import VoxelModel

# Create new model
model = VoxelModel(size=(32, 32, 32))

# Set/get voxels
model.set_voxel(x, y, z, color_index)
color = model.get_voxel(x, y, z)

# Operations
model.fill_region(start, end, color_index)
model.flood_fill(start, color_index)
model.rotate_90('y')
model.flip('x')
model.hollow(thickness=1)

# Undo/Redo
model.undo()
model.redo()
```

### VoxelPalette

```python
from voxedit.core import VoxelPalette

# Create palette
palette = VoxelPalette()

# Get/set colors
color = palette.get_color(index)
palette.set_color(index, r, g, b, a)

# Find nearest color
index = palette.find_nearest_color(r, g, b)

# Load/save
palette.load_from_file("palette.pal")
palette.save_to_file("palette.png")
```

### VoxelOperations

```python
from voxedit.core import VoxelModel, VoxelOperations

model = VoxelModel(size=(64, 64, 64))
ops = VoxelOperations(model)

# Drawing
ops.draw_sphere(center=(32, 32, 32), radius=10, value=1)
ops.draw_box(start=(10, 10, 10), end=(20, 20, 20), value=2)
ops.draw_line(start=(0, 0, 0), end=(63, 63, 63), value=3)

# Transformations
ops.rotate('y', 90)
ops.mirror('x')
ops.shell(thickness=2)
ops.smooth(iterations=1)
```

## Project Structure

```
voxedit/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── voxedit/
│   ├── __init__.py        # Package initialization
│   ├── core/
│   │   ├── __init__.py
│   │   ├── voxel_model.py # Core voxel data structure
│   │   ├── palette.py     # Color palette management
│   │   └── operations.py  # Voxel manipulation operations
│   ├── formats/
│   │   ├── __init__.py    # Format manager
│   │   ├── vox.py         # MagicaVoxel format
│   │   ├── minecraft.py   # Minecraft formats
│   │   ├── mesh.py        # Mesh exports (OBJ, STL, etc.)
│   │   └── voxel.py       # Other voxel formats
│   └── gui/
│       ├── __init__.py
│       ├── main_window.py # Main application window
│       ├── viewport.py    # OpenGL 3D viewport
│       ├── tool_panel.py  # Editing tools panel
│       └── palette_panel.py # Palette management panel
└── docs/
    └── screenshot.png
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MagicaVoxel for the .vox format specification
- PyQt team for the excellent GUI framework
- NumPy and SciPy communities for scientific computing tools
- Teardown community for inspiration

## Support

- **Issues**: Report bugs or request features on GitHub Issues
- **Discussions**: Join the community discussions
- **Wiki**: Check the wiki for detailed documentation

---

**VoxEdit** - Create amazing voxel art! 🎨
