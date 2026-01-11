"""
VoxelModel - Core Voxel Data Structure
======================================

Provides the fundamental voxel grid storage and manipulation capabilities.
Uses numpy for efficient memory usage and fast operations.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class VoxelModel:
    """
    Core voxel model representation using numpy arrays.
    
    Attributes:
        voxels: 3D numpy array of voxel indices (0 = empty, 1-255 = palette index)
        size: Tuple of (x, y, z) dimensions
        palette: Reference to the color palette
        name: Model name for identification
        position: Position offset in world space
    """
    
    size: Tuple[int, int, int] = (32, 32, 32)
    name: str = "Untitled"
    position: Tuple[int, int, int] = (0, 0, 0)
    _voxels: np.ndarray = field(default=None, repr=False)
    _history: List[np.ndarray] = field(default_factory=list, repr=False)
    _history_index: int = field(default=-1, repr=False)
    _max_history: int = field(default=50, repr=False)
    
    def __post_init__(self):
        """Initialize the voxel array if not provided."""
        if self._voxels is None:
            self._voxels = np.zeros(self.size, dtype=np.uint8)
        self._save_state()
    
    @property
    def voxels(self) -> np.ndarray:
        """Get the voxel data array."""
        return self._voxels
    
    @voxels.setter
    def voxels(self, value: np.ndarray):
        """Set the voxel data array."""
        self._voxels = value.astype(np.uint8)
        self.size = value.shape
    
    @classmethod
    def create(cls, size_x: int = 32, size_y: int = 32, size_z: int = 32, 
               name: str = "Untitled") -> 'VoxelModel':
        """
        Factory method to create a new empty voxel model.
        
        Args:
            size_x: X dimension
            size_y: Y dimension
            size_z: Z dimension
            name: Model name
            
        Returns:
            New VoxelModel instance
        """
        model = cls(size=(size_x, size_y, size_z), name=name)
        return model
    
    @classmethod
    def from_array(cls, array: np.ndarray, name: str = "Imported") -> 'VoxelModel':
        """
        Create a VoxelModel from an existing numpy array.
        
        Args:
            array: 3D numpy array of voxel data
            name: Model name
            
        Returns:
            New VoxelModel instance
        """
        model = cls(size=array.shape, name=name)
        model._voxels = array.astype(np.uint8)
        return model
    
    def _save_state(self):
        """Save current state to history for undo functionality."""
        # Remove any future states if we're not at the end
        if self._history_index < len(self._history) - 1:
            self._history = self._history[:self._history_index + 1]
        
        # Add current state
        self._history.append(self._voxels.copy())
        self._history_index = len(self._history) - 1
        
        # Limit history size
        if len(self._history) > self._max_history:
            self._history.pop(0)
            self._history_index -= 1
    
    def undo(self) -> bool:
        """
        Undo the last operation.
        
        Returns:
            True if undo was successful, False otherwise
        """
        if self._history_index > 0:
            self._history_index -= 1
            self._voxels = self._history[self._history_index].copy()
            return True
        return False
    
    def redo(self) -> bool:
        """
        Redo the last undone operation.
        
        Returns:
            True if redo was successful, False otherwise
        """
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self._voxels = self._history[self._history_index].copy()
            return True
        return False
    
    def commit(self):
        """Commit current state to history (call after completing an operation)."""
        self._save_state()
    
    def get_voxel(self, x: int, y: int, z: int) -> int:
        """
        Get the voxel value at the specified position.
        
        Args:
            x, y, z: Voxel coordinates
            
        Returns:
            Voxel palette index (0 = empty)
        """
        if self.is_valid_position(x, y, z):
            return int(self._voxels[x, y, z])
        return 0
    
    def set_voxel(self, x: int, y: int, z: int, value: int):
        """
        Set the voxel value at the specified position.
        
        Args:
            x, y, z: Voxel coordinates
            value: Palette index (0 = empty, 1-255 = color)
        """
        if self.is_valid_position(x, y, z):
            self._voxels[x, y, z] = np.uint8(value)
    
    def is_valid_position(self, x: int, y: int, z: int) -> bool:
        """Check if coordinates are within the model bounds."""
        return (0 <= x < self.size[0] and 
                0 <= y < self.size[1] and 
                0 <= z < self.size[2])
    
    def get_bounds(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Get the bounding box of non-empty voxels.
        
        Returns:
            Tuple of (min_corner, max_corner)
        """
        non_empty = np.argwhere(self._voxels > 0)
        if len(non_empty) == 0:
            return ((0, 0, 0), (0, 0, 0))
        
        min_corner = tuple(non_empty.min(axis=0))
        max_corner = tuple(non_empty.max(axis=0))
        return (min_corner, max_corner)
    
    def get_voxel_count(self) -> int:
        """Get the number of non-empty voxels."""
        return int(np.count_nonzero(self._voxels))
    
    def clear(self):
        """Clear all voxels in the model."""
        self._voxels.fill(0)
        self.commit()
    
    def fill(self, value: int):
        """Fill the entire model with a single voxel value."""
        self._voxels.fill(np.uint8(value))
        self.commit()
    
    def fill_region(self, start: Tuple[int, int, int], end: Tuple[int, int, int], 
                    value: int):
        """
        Fill a rectangular region with a voxel value.
        
        Args:
            start: Starting corner (x, y, z)
            end: Ending corner (x, y, z)
            value: Palette index to fill with
        """
        x1, y1, z1 = max(0, min(start[0], end[0])), max(0, min(start[1], end[1])), max(0, min(start[2], end[2]))
        x2, y2, z2 = min(self.size[0], max(start[0], end[0]) + 1), min(self.size[1], max(start[1], end[1]) + 1), min(self.size[2], max(start[2], end[2]) + 1)
        
        self._voxels[x1:x2, y1:y2, z1:z2] = np.uint8(value)
        self.commit()
    
    def copy_region(self, start: Tuple[int, int, int], 
                    end: Tuple[int, int, int]) -> np.ndarray:
        """
        Copy a rectangular region of voxels.
        
        Args:
            start: Starting corner
            end: Ending corner
            
        Returns:
            Numpy array containing the copied voxels
        """
        x1, y1, z1 = max(0, min(start[0], end[0])), max(0, min(start[1], end[1])), max(0, min(start[2], end[2]))
        x2, y2, z2 = min(self.size[0], max(start[0], end[0]) + 1), min(self.size[1], max(start[1], end[1]) + 1), min(self.size[2], max(start[2], end[2]) + 1)
        
        return self._voxels[x1:x2, y1:y2, z1:z2].copy()
    
    def paste_region(self, data: np.ndarray, position: Tuple[int, int, int], 
                     overwrite: bool = True):
        """
        Paste a voxel array at the specified position.
        
        Args:
            data: Numpy array of voxel data to paste
            position: Target position (x, y, z)
            overwrite: If True, overwrite existing voxels; if False, only fill empty
        """
        px, py, pz = position
        dx, dy, dz = data.shape
        
        # Calculate valid region
        x1, y1, z1 = max(0, px), max(0, py), max(0, pz)
        x2 = min(self.size[0], px + dx)
        y2 = min(self.size[1], py + dy)
        z2 = min(self.size[2], pz + dz)
        
        # Calculate source offsets
        sx1 = x1 - px
        sy1 = y1 - py
        sz1 = z1 - pz
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)
        sz2 = sz1 + (z2 - z1)
        
        if overwrite:
            self._voxels[x1:x2, y1:y2, z1:z2] = data[sx1:sx2, sy1:sy2, sz1:sz2]
        else:
            # Only paste where destination is empty
            mask = self._voxels[x1:x2, y1:y2, z1:z2] == 0
            target = self._voxels[x1:x2, y1:y2, z1:z2]
            source = data[sx1:sx2, sy1:sy2, sz1:sz2]
            target[mask] = source[mask]
        
        self.commit()
    
    def resize(self, new_size: Tuple[int, int, int], anchor: str = "center"):
        """
        Resize the voxel model.
        
        Args:
            new_size: New dimensions (x, y, z)
            anchor: Where to anchor the existing content 
                    ("center", "origin", "corner")
        """
        new_voxels = np.zeros(new_size, dtype=np.uint8)
        
        # Calculate offsets based on anchor
        if anchor == "center":
            offset = tuple((new_size[i] - self.size[i]) // 2 for i in range(3))
        elif anchor == "origin":
            offset = (0, 0, 0)
        else:  # corner
            offset = tuple(max(0, new_size[i] - self.size[i]) for i in range(3))
        
        # Copy existing data
        for x in range(min(self.size[0], new_size[0])):
            for y in range(min(self.size[1], new_size[1])):
                for z in range(min(self.size[2], new_size[2])):
                    nx = x + offset[0]
                    ny = y + offset[1]
                    nz = z + offset[2]
                    if 0 <= nx < new_size[0] and 0 <= ny < new_size[1] and 0 <= nz < new_size[2]:
                        if x < self.size[0] and y < self.size[1] and z < self.size[2]:
                            new_voxels[nx, ny, nz] = self._voxels[x, y, z]
        
        self._voxels = new_voxels
        self.size = new_size
        self.commit()
    
    def rotate_90(self, axis: str = 'y'):
        """
        Rotate the model 90 degrees around an axis.
        
        Args:
            axis: Rotation axis ('x', 'y', or 'z')
        """
        if axis == 'x':
            self._voxels = np.rot90(self._voxels, axes=(1, 2))
        elif axis == 'y':
            self._voxels = np.rot90(self._voxels, axes=(0, 2))
        elif axis == 'z':
            self._voxels = np.rot90(self._voxels, axes=(0, 1))
        
        self.size = self._voxels.shape
        self.commit()
    
    def flip(self, axis: str = 'x'):
        """
        Flip the model along an axis.
        
        Args:
            axis: Flip axis ('x', 'y', or 'z')
        """
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        self._voxels = np.flip(self._voxels, axis=axis_map[axis])
        self.commit()
    
    def mirror(self, axis: str = 'x', positive: bool = True):
        """
        Mirror the model, copying one half to the other.
        
        Args:
            axis: Mirror axis ('x', 'y', or 'z')
            positive: If True, copy positive half to negative; vice versa
        """
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        mid = self.size[axis_idx] // 2
        
        if axis == 'x':
            if positive:
                self._voxels[:mid, :, :] = np.flip(self._voxels[mid:mid*2, :, :], axis=0)
            else:
                self._voxels[mid:, :, :] = np.flip(self._voxels[:mid, :, :], axis=0)
        elif axis == 'y':
            if positive:
                self._voxels[:, :mid, :] = np.flip(self._voxels[:, mid:mid*2, :], axis=1)
            else:
                self._voxels[:, mid:, :] = np.flip(self._voxels[:, :mid, :], axis=1)
        elif axis == 'z':
            if positive:
                self._voxels[:, :, :mid] = np.flip(self._voxels[:, :, mid:mid*2], axis=2)
            else:
                self._voxels[:, :, mid:] = np.flip(self._voxels[:, :, :mid], axis=2)
        
        self.commit()
    
    def get_surface_voxels(self) -> List[Tuple[int, int, int, int]]:
        """
        Get all voxels that are on the surface (have at least one empty neighbor).
        
        Returns:
            List of (x, y, z, palette_index) tuples
        """
        surface = []
        neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                for z in range(self.size[2]):
                    if self._voxels[x, y, z] > 0:
                        # Check if any neighbor is empty or outside bounds
                        is_surface = False
                        for dx, dy, dz in neighbors:
                            nx, ny, nz = x + dx, y + dy, z + dz
                            if not self.is_valid_position(nx, ny, nz) or self._voxels[nx, ny, nz] == 0:
                                is_surface = True
                                break
                        if is_surface:
                            surface.append((x, y, z, int(self._voxels[x, y, z])))
        
        return surface
    
    def get_all_voxels(self) -> List[Tuple[int, int, int, int]]:
        """
        Get all non-empty voxels.
        
        Returns:
            List of (x, y, z, palette_index) tuples
        """
        non_empty = np.argwhere(self._voxels > 0)
        return [(int(x), int(y), int(z), int(self._voxels[x, y, z])) 
                for x, y, z in non_empty]
    
    def flood_fill(self, start: Tuple[int, int, int], value: int, 
                   replace_only: Optional[int] = None):
        """
        Flood fill from a starting point.
        
        Args:
            start: Starting position (x, y, z)
            value: New voxel value to fill with
            replace_only: If set, only replace voxels with this value
        """
        if not self.is_valid_position(*start):
            return
        
        target = self._voxels[start[0], start[1], start[2]]
        if replace_only is not None and target != replace_only:
            return
        if target == value:
            return
        
        stack = [start]
        visited = set()
        
        while stack:
            x, y, z = stack.pop()
            
            if (x, y, z) in visited:
                continue
            if not self.is_valid_position(x, y, z):
                continue
            if self._voxels[x, y, z] != target:
                continue
            
            visited.add((x, y, z))
            self._voxels[x, y, z] = np.uint8(value)
            
            # Add neighbors
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                stack.append((x + dx, y + dy, z + dz))
        
        self.commit()
    
    def hollow(self, wall_thickness: int = 1):
        """
        Hollow out the model, keeping only a shell.
        
        Args:
            wall_thickness: Thickness of the walls to keep
        """
        from scipy import ndimage
        
        # Create interior mask (eroded from the filled region)
        filled = self._voxels > 0
        interior = ndimage.binary_erosion(filled, iterations=wall_thickness)
        
        # Clear interior voxels
        self._voxels[interior] = 0
        self.commit()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the model to a dictionary."""
        return {
            'name': self.name,
            'size': self.size,
            'position': self.position,
            'voxels': self._voxels.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoxelModel':
        """Deserialize a model from a dictionary."""
        model = cls(
            size=tuple(data['size']),
            name=data.get('name', 'Imported'),
            position=tuple(data.get('position', (0, 0, 0)))
        )
        model._voxels = np.array(data['voxels'], dtype=np.uint8)
        return model
    
    def clone(self) -> 'VoxelModel':
        """Create a deep copy of this model."""
        new_model = VoxelModel(
            size=self.size,
            name=f"{self.name}_copy",
            position=self.position
        )
        new_model._voxels = self._voxels.copy()
        return new_model
