"""
VoxelOperations - Advanced Voxel Manipulation Operations
=========================================================

Provides high-level operations for voxel manipulation including
noise generation, shape primitives, and algorithmic modifications.
"""

import numpy as np
from typing import Tuple, Optional, List, Callable, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from voxedit.core.voxel_model import VoxelModel


class VoxelOperations:
    """
    Operations class for manipulating a voxel model.
    
    Can be used both as instance methods (with a model) or static methods.
    """
    
    def __init__(self, model: 'VoxelModel' = None):
        """
        Initialize operations with an optional model reference.
        
        Args:
            model: VoxelModel to operate on
        """
        self.model = model
    
    def rotate(self, axis: str, degrees: int):
        """
        Rotate the model around an axis.
        
        Args:
            axis: Rotation axis ('x', 'y', or 'z')
            degrees: Rotation angle (must be multiple of 90)
        """
        if self.model is None:
            return
        
        rotations = (degrees % 360) // 90
        for _ in range(rotations):
            self.model.rotate_90(axis)
    
    def mirror(self, axis: str):
        """
        Mirror the model along an axis.
        
        Args:
            axis: Mirror axis ('x', 'y', or 'z')
        """
        if self.model is None:
            return
        self.model.flip(axis)
        self.model.commit()
    
    def shell(self, thickness: int = 1):
        """
        Create a hollow shell of the model.
        
        Args:
            thickness: Shell wall thickness
        """
        if self.model is None:
            return
        self.model.hollow(thickness)
    
    def smooth(self, iterations: int = 1):
        """
        Smooth the model surface.
        
        Args:
            iterations: Number of smoothing iterations
        """
        if self.model is None:
            return
        
        for _ in range(iterations):
            self.model._voxels = VoxelOperations.smooth_voxels(
                self.model._voxels, threshold=4
            )
        self.model.commit()
    
    def erode(self, iterations: int = 1):
        """
        Erode the model surface.
        
        Args:
            iterations: Number of erosion iterations
        """
        if self.model is None:
            return
        
        self.model._voxels = VoxelOperations.erode_voxels(
            self.model._voxels, iterations
        )
        self.model.commit()
    
    def dilate(self, value: int = 1, iterations: int = 1):
        """
        Dilate/grow the model surface.
        
        Args:
            value: Voxel value for new voxels
            iterations: Number of dilation iterations
        """
        if self.model is None:
            return
        
        self.model._voxels = VoxelOperations.dilate_voxels(
            self.model._voxels, value, iterations
        )
        self.model.commit()
    
    def draw_line(self, start: Tuple[int, int, int], end: Tuple[int, int, int], 
                  value: int = 1):
        """
        Draw a line of voxels between two points.
        
        Args:
            start: Start point (x, y, z)
            end: End point (x, y, z)
            value: Voxel value
        """
        if self.model is None:
            return
        
        # Bresenham's 3D line algorithm
        x1, y1, z1 = start
        x2, y2, z2 = end
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        sz = 1 if z1 < z2 else -1
        
        dm = max(dx, dy, dz)
        
        x, y, z = x1, y1, z1
        
        if dm == 0:
            self.model.set_voxel(x, y, z, value)
            return
        
        for _ in range(dm + 1):
            self.model.set_voxel(x, y, z, value)
            
            if x == x2 and y == y2 and z == z2:
                break
            
            # Move towards target
            if dm == dx:
                x += sx
                if dy * abs(x - x1) >= dx * abs(y - y1) + dx // 2:
                    y += sy if (y2 - y1) * (x - x1) > (x2 - x1) * (y - y1) else 0
                if dz * abs(x - x1) >= dx * abs(z - z1) + dx // 2:
                    z += sz if (z2 - z1) * (x - x1) > (x2 - x1) * (z - z1) else 0
            elif dm == dy:
                y += sy
                if dx * abs(y - y1) >= dy * abs(x - x1) + dy // 2:
                    x += sx if (x2 - x1) * (y - y1) > (y2 - y1) * (x - x1) else 0
                if dz * abs(y - y1) >= dy * abs(z - z1) + dy // 2:
                    z += sz if (z2 - z1) * (y - y1) > (y2 - y1) * (z - z1) else 0
            else:
                z += sz
                if dx * abs(z - z1) >= dz * abs(x - x1) + dz // 2:
                    x += sx if (x2 - x1) * (z - z1) > (z2 - z1) * (x - x1) else 0
                if dy * abs(z - z1) >= dz * abs(y - y1) + dz // 2:
                    y += sy if (y2 - y1) * (z - z1) > (z2 - z1) * (y - y1) else 0
        
        self.model.commit()
    
    def draw_sphere(self, center: Tuple[int, int, int], radius: int, value: int = 1):
        """
        Draw a sphere of voxels.
        
        Args:
            center: Center point (x, y, z)
            radius: Sphere radius
            value: Voxel value
        """
        if self.model is None:
            return
        
        cx, cy, cz = center
        r2 = radius * radius
        
        for x in range(cx - radius, cx + radius + 1):
            for y in range(cy - radius, cy + radius + 1):
                for z in range(cz - radius, cz + radius + 1):
                    if self.model.is_valid_position(x, y, z):
                        dist2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                        if dist2 <= r2:
                            self.model.set_voxel(x, y, z, value)
        
        self.model.commit()
    
    def draw_box(self, start: Tuple[int, int, int], end: Tuple[int, int, int],
                 value: int = 1, filled: bool = True):
        """
        Draw a box of voxels.
        
        Args:
            start: First corner (x, y, z)
            end: Opposite corner (x, y, z)
            value: Voxel value
            filled: If True, solid box; if False, hollow
        """
        if self.model is None:
            return
        
        x1, y1, z1 = min(start[0], end[0]), min(start[1], end[1]), min(start[2], end[2])
        x2, y2, z2 = max(start[0], end[0]), max(start[1], end[1]), max(start[2], end[2])
        
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                for z in range(z1, z2 + 1):
                    if self.model.is_valid_position(x, y, z):
                        if filled:
                            self.model.set_voxel(x, y, z, value)
                        else:
                            # Only draw faces
                            if x == x1 or x == x2 or y == y1 or y == y2 or z == z1 or z == z2:
                                self.model.set_voxel(x, y, z, value)
        
        self.model.commit()
    
    def extrude(self, axis: str, distance: int, value: int = None):
        """
        Extrude existing voxels along an axis.
        
        Args:
            axis: Extrusion axis ('x', 'y', or 'z')
            distance: Extrusion distance (positive or negative)
            value: Optional value override
        """
        if self.model is None:
            return
        
        direction = 1 if distance > 0 else -1
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        
        # Get surface voxels on the extrusion face
        surface = self.model.get_surface_voxels()
        
        for x, y, z, v in surface:
            pos = [x, y, z]
            
            for d in range(1, abs(distance) + 1):
                new_pos = list(pos)
                new_pos[axis_idx] += d * direction
                
                if self.model.is_valid_position(*new_pos):
                    self.model.set_voxel(*new_pos, value if value else v)
        
        self.model.commit()
    
    # ==================== Static Methods ====================
    
    @staticmethod
    def smooth_voxels(voxels: np.ndarray, threshold: int = 4) -> np.ndarray:
        """
        Smooth voxels using cellular automata.
        
        Args:
            voxels: Input voxel array
            threshold: Minimum neighbors to keep/create voxel
            
        Returns:
            Smoothed voxel array
        """
        from scipy import ndimage
        
        result = np.zeros_like(voxels)
        mask = voxels > 0
        
        kernel = np.ones((3, 3, 3), dtype=np.int32)
        kernel[1, 1, 1] = 0
        
        neighbor_count = ndimage.convolve(mask.astype(np.int32), kernel, mode='constant')
        result[neighbor_count >= threshold] = 1
        result[result > 0] = voxels[result > 0]
        
        return result
    
    @staticmethod
    def erode_voxels(voxels: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        Erode voxels (remove surface layer).
        
        Args:
            voxels: Input voxel array
            iterations: Number of erosion iterations
            
        Returns:
            Eroded voxel array
        """
        from scipy import ndimage
        
        result = voxels.copy()
        mask = result > 0
        eroded = ndimage.binary_erosion(mask, iterations=iterations)
        result[~eroded] = 0
        
        return result
    
    @staticmethod
    def dilate_voxels(voxels: np.ndarray, value: int = 1, 
                      iterations: int = 1) -> np.ndarray:
        """
        Dilate voxels (grow surface).
        
        Args:
            voxels: Input voxel array
            value: Value for new voxels
            iterations: Number of dilation iterations
            
        Returns:
            Dilated voxel array
        """
        from scipy import ndimage
        
        result = voxels.copy()
        mask = result > 0
        dilated = ndimage.binary_dilation(mask, iterations=iterations)
        result[dilated & (result == 0)] = value
        
        return result
    
    @staticmethod
    def create_sphere(radius: float, center: Tuple[float, float, float] = None,
                      size: Tuple[int, int, int] = None, 
                      value: int = 1, hollow: bool = False,
                      wall_thickness: float = 1.0) -> np.ndarray:
        """
        Create a sphere of voxels.
        
        Args:
            radius: Sphere radius
            center: Center point (defaults to center of array)
            size: Output array size (defaults to fit sphere)
            value: Voxel value
            hollow: If True, create hollow sphere
            wall_thickness: Wall thickness for hollow sphere
            
        Returns:
            3D numpy array with the sphere
        """
        if size is None:
            size = (int(radius * 2) + 2, int(radius * 2) + 2, int(radius * 2) + 2)
        
        if center is None:
            center = (size[0] / 2, size[1] / 2, size[2] / 2)
        
        result = np.zeros(size, dtype=np.uint8)
        
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    dist = math.sqrt(
                        (x - center[0]) ** 2 +
                        (y - center[1]) ** 2 +
                        (z - center[2]) ** 2
                    )
                    
                    if hollow:
                        if radius - wall_thickness <= dist <= radius:
                            result[x, y, z] = value
                    else:
                        if dist <= radius:
                            result[x, y, z] = value
        
        return result
    
    @staticmethod
    def create_box(dimensions: Tuple[int, int, int],
                   position: Tuple[int, int, int] = (0, 0, 0),
                   size: Tuple[int, int, int] = None,
                   value: int = 1, hollow: bool = False,
                   wall_thickness: int = 1) -> np.ndarray:
        """
        Create a box of voxels.
        
        Args:
            dimensions: Box dimensions (width, height, depth)
            position: Position offset
            size: Output array size
            value: Voxel value
            hollow: If True, create hollow box
            wall_thickness: Wall thickness for hollow box
            
        Returns:
            3D numpy array with the box
        """
        if size is None:
            size = tuple(d + p for d, p in zip(dimensions, position))
        
        result = np.zeros(size, dtype=np.uint8)
        
        x1, y1, z1 = position
        x2 = min(size[0], x1 + dimensions[0])
        y2 = min(size[1], y1 + dimensions[1])
        z2 = min(size[2], z1 + dimensions[2])
        
        if hollow:
            # Outer box
            result[x1:x2, y1:y2, z1:z2] = value
            # Inner cavity
            inner_x1 = x1 + wall_thickness
            inner_y1 = y1 + wall_thickness
            inner_z1 = z1 + wall_thickness
            inner_x2 = x2 - wall_thickness
            inner_y2 = y2 - wall_thickness
            inner_z2 = z2 - wall_thickness
            if inner_x2 > inner_x1 and inner_y2 > inner_y1 and inner_z2 > inner_z1:
                result[inner_x1:inner_x2, inner_y1:inner_y2, inner_z1:inner_z2] = 0
        else:
            result[x1:x2, y1:y2, z1:z2] = value
        
        return result
    
    @staticmethod
    def create_cylinder(radius: float, height: int,
                        axis: str = 'y',
                        center: Tuple[float, float, float] = None,
                        size: Tuple[int, int, int] = None,
                        value: int = 1, hollow: bool = False,
                        wall_thickness: float = 1.0) -> np.ndarray:
        """
        Create a cylinder of voxels.
        
        Args:
            radius: Cylinder radius
            height: Cylinder height
            axis: Axis to align cylinder ('x', 'y', or 'z')
            center: Center of the base
            size: Output array size
            value: Voxel value
            hollow: If True, create hollow cylinder
            wall_thickness: Wall thickness for hollow cylinder
            
        Returns:
            3D numpy array with the cylinder
        """
        if size is None:
            if axis == 'y':
                size = (int(radius * 2) + 2, height + 1, int(radius * 2) + 2)
            elif axis == 'x':
                size = (height + 1, int(radius * 2) + 2, int(radius * 2) + 2)
            else:
                size = (int(radius * 2) + 2, int(radius * 2) + 2, height + 1)
        
        if center is None:
            if axis == 'y':
                center = (size[0] / 2, 0, size[2] / 2)
            elif axis == 'x':
                center = (0, size[1] / 2, size[2] / 2)
            else:
                center = (size[0] / 2, size[1] / 2, 0)
        
        result = np.zeros(size, dtype=np.uint8)
        
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    if axis == 'y':
                        dist = math.sqrt((x - center[0]) ** 2 + (z - center[2]) ** 2)
                        in_height = 0 <= y < height
                    elif axis == 'x':
                        dist = math.sqrt((y - center[1]) ** 2 + (z - center[2]) ** 2)
                        in_height = 0 <= x < height
                    else:
                        dist = math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                        in_height = 0 <= z < height
                    
                    if in_height:
                        if hollow:
                            if radius - wall_thickness <= dist <= radius:
                                result[x, y, z] = value
                        else:
                            if dist <= radius:
                                result[x, y, z] = value
        
        return result
    
    @staticmethod
    def create_pyramid(base_size: int, height: int,
                       position: Tuple[int, int, int] = (0, 0, 0),
                       size: Tuple[int, int, int] = None,
                       value: int = 1) -> np.ndarray:
        """
        Create a pyramid of voxels.
        
        Args:
            base_size: Size of the square base
            height: Pyramid height
            position: Position offset
            size: Output array size
            value: Voxel value
            
        Returns:
            3D numpy array with the pyramid
        """
        if size is None:
            size = (base_size + position[0], height + position[1], base_size + position[2])
        
        result = np.zeros(size, dtype=np.uint8)
        
        for y in range(height):
            # Calculate layer size based on height
            layer_size = int(base_size * (height - y) / height)
            offset = (base_size - layer_size) // 2
            
            x1 = position[0] + offset
            z1 = position[2] + offset
            x2 = min(size[0], x1 + layer_size)
            z2 = min(size[2], z1 + layer_size)
            
            result[x1:x2, position[1] + y, z1:z2] = value
        
        return result
    
    @staticmethod
    def create_cone(radius: float, height: int,
                    center: Tuple[float, float, float] = None,
                    size: Tuple[int, int, int] = None,
                    value: int = 1, inverted: bool = False) -> np.ndarray:
        """
        Create a cone of voxels.
        
        Args:
            radius: Base radius
            height: Cone height
            center: Center of the base
            size: Output array size
            value: Voxel value
            inverted: If True, point faces down
            
        Returns:
            3D numpy array with the cone
        """
        if size is None:
            dim = int(radius * 2) + 2
            size = (dim, height + 1, dim)
        
        if center is None:
            center = (size[0] / 2, 0, size[2] / 2)
        
        result = np.zeros(size, dtype=np.uint8)
        
        for x in range(size[0]):
            for y in range(height):
                for z in range(size[2]):
                    if inverted:
                        layer_radius = radius * y / height
                    else:
                        layer_radius = radius * (height - y) / height
                    
                    dist = math.sqrt((x - center[0]) ** 2 + (z - center[2]) ** 2)
                    
                    if dist <= layer_radius:
                        result[x, y, z] = value
        
        return result
    
    @staticmethod
    def create_torus(major_radius: float, minor_radius: float,
                     center: Tuple[float, float, float] = None,
                     size: Tuple[int, int, int] = None,
                     value: int = 1) -> np.ndarray:
        """
        Create a torus (donut shape) of voxels.
        
        Args:
            major_radius: Distance from center to tube center
            minor_radius: Radius of the tube
            center: Center of the torus
            size: Output array size
            value: Voxel value
            
        Returns:
            3D numpy array with the torus
        """
        if size is None:
            dim = int((major_radius + minor_radius) * 2) + 2
            size = (dim, int(minor_radius * 2) + 2, dim)
        
        if center is None:
            center = (size[0] / 2, size[1] / 2, size[2] / 2)
        
        result = np.zeros(size, dtype=np.uint8)
        
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    # Distance from y-axis (center line)
                    dist_xz = math.sqrt((x - center[0]) ** 2 + (z - center[2]) ** 2)
                    
                    # Distance from the torus ring
                    dist = math.sqrt((dist_xz - major_radius) ** 2 + (y - center[1]) ** 2)
                    
                    if dist <= minor_radius:
                        result[x, y, z] = value
        
        return result
    
    @staticmethod
    def apply_noise(voxels: np.ndarray, noise_type: str = 'perlin',
                    scale: float = 0.1, threshold: float = 0.5,
                    value: int = 1, seed: int = None) -> np.ndarray:
        """
        Apply noise to voxel array.
        
        Args:
            voxels: Input voxel array
            noise_type: Type of noise ('perlin', 'simplex', 'random')
            scale: Noise scale
            threshold: Threshold for voxel generation
            value: Voxel value
            seed: Random seed
            
        Returns:
            Modified voxel array
        """
        if seed is not None:
            np.random.seed(seed)
        
        result = voxels.copy()
        size = voxels.shape
        
        if noise_type == 'random':
            noise = np.random.random(size)
        else:
            # Simple perlin-like noise using sine waves
            noise = np.zeros(size)
            for x in range(size[0]):
                for y in range(size[1]):
                    for z in range(size[2]):
                        noise[x, y, z] = (
                            math.sin(x * scale) +
                            math.sin(y * scale * 1.3) +
                            math.sin(z * scale * 0.7) +
                            math.sin((x + y) * scale * 0.5) +
                            math.sin((y + z) * scale * 0.4) +
                            math.sin((x + z) * scale * 0.6)
                        ) / 6.0 + 0.5
        
        # Apply threshold
        mask = noise > threshold
        result[mask] = value
        
        return result
    
    @staticmethod
    def erode(voxels: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        Erode voxels (remove surface layer).
        
        Args:
            voxels: Input voxel array
            iterations: Number of erosion iterations
            
        Returns:
            Eroded voxel array
        """
        from scipy import ndimage
        
        result = voxels.copy()
        mask = result > 0
        eroded = ndimage.binary_erosion(mask, iterations=iterations)
        result[~eroded] = 0
        
        return result
    
    @staticmethod
    def dilate(voxels: np.ndarray, value: int = 1, 
               iterations: int = 1) -> np.ndarray:
        """
        Dilate voxels (grow surface).
        
        Args:
            voxels: Input voxel array
            value: Value for new voxels
            iterations: Number of dilation iterations
            
        Returns:
            Dilated voxel array
        """
        from scipy import ndimage
        
        result = voxels.copy()
        mask = result > 0
        dilated = ndimage.binary_dilation(mask, iterations=iterations)
        result[dilated & (result == 0)] = value
        
        return result
    
    @staticmethod
    def smooth(voxels: np.ndarray, threshold: int = 4) -> np.ndarray:
        """
        Smooth voxels using cellular automata.
        
        Args:
            voxels: Input voxel array
            threshold: Minimum neighbors to keep/create voxel
            
        Returns:
            Smoothed voxel array
        """
        from scipy import ndimage
        
        result = np.zeros_like(voxels)
        mask = voxels > 0
        
        # Count neighbors for each voxel
        kernel = np.ones((3, 3, 3), dtype=np.int32)
        kernel[1, 1, 1] = 0
        
        neighbor_count = ndimage.convolve(mask.astype(np.int32), kernel, mode='constant')
        
        # Apply threshold
        result[neighbor_count >= threshold] = 1
        
        # Preserve original values where keeping voxels
        result[result > 0] = voxels[result > 0]
        
        return result
    
    @staticmethod
    def boolean_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Boolean union of two voxel arrays.
        
        Args:
            a: First voxel array
            b: Second voxel array (will be cropped/padded to match a)
            
        Returns:
            Union of both arrays
        """
        # Resize b to match a if needed
        if a.shape != b.shape:
            result = np.zeros_like(a)
            min_shape = tuple(min(a.shape[i], b.shape[i]) for i in range(3))
            result[:min_shape[0], :min_shape[1], :min_shape[2]] = \
                a[:min_shape[0], :min_shape[1], :min_shape[2]]
            
            b_cropped = b[:min_shape[0], :min_shape[1], :min_shape[2]]
            mask = b_cropped > 0
            result[:min_shape[0], :min_shape[1], :min_shape[2]][mask] = b_cropped[mask]
            return result
        else:
            result = a.copy()
            mask = b > 0
            result[mask] = b[mask]
            return result
    
    @staticmethod
    def boolean_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Boolean subtraction (a - b).
        
        Args:
            a: Array to subtract from
            b: Array to subtract
            
        Returns:
            Difference of arrays
        """
        result = a.copy()
        
        min_shape = tuple(min(a.shape[i], b.shape[i]) for i in range(3))
        b_cropped = b[:min_shape[0], :min_shape[1], :min_shape[2]]
        mask = b_cropped > 0
        result[:min_shape[0], :min_shape[1], :min_shape[2]][mask] = 0
        
        return result
    
    @staticmethod
    def boolean_intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Boolean intersection of two voxel arrays.
        
        Args:
            a: First voxel array
            b: Second voxel array
            
        Returns:
            Intersection of both arrays
        """
        result = np.zeros_like(a)
        
        min_shape = tuple(min(a.shape[i], b.shape[i]) for i in range(3))
        
        a_cropped = a[:min_shape[0], :min_shape[1], :min_shape[2]]
        b_cropped = b[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        mask = (a_cropped > 0) & (b_cropped > 0)
        result[:min_shape[0], :min_shape[1], :min_shape[2]][mask] = \
            a_cropped[mask]
        
        return result
    
    @staticmethod
    def extrude(voxels_2d: np.ndarray, height: int, 
                value: int = 1, axis: str = 'y') -> np.ndarray:
        """
        Extrude a 2D pattern into 3D.
        
        Args:
            voxels_2d: 2D numpy array
            height: Extrusion height
            value: Voxel value
            axis: Axis to extrude along ('x', 'y', or 'z')
            
        Returns:
            3D extruded voxel array
        """
        if axis == 'y':
            result = np.zeros((voxels_2d.shape[0], height, voxels_2d.shape[1]), 
                            dtype=np.uint8)
            for y in range(height):
                result[:, y, :] = voxels_2d
        elif axis == 'x':
            result = np.zeros((height, voxels_2d.shape[0], voxels_2d.shape[1]), 
                            dtype=np.uint8)
            for x in range(height):
                result[x, :, :] = voxels_2d
        else:  # z
            result = np.zeros((voxels_2d.shape[0], voxels_2d.shape[1], height), 
                            dtype=np.uint8)
            for z in range(height):
                result[:, :, z] = voxels_2d
        
        return result
    
    @staticmethod
    def revolve(voxels_2d: np.ndarray, segments: int = 16,
                axis: str = 'y') -> np.ndarray:
        """
        Revolve a 2D profile around an axis.
        
        Args:
            voxels_2d: 2D profile (XY plane, revolved around Y axis)
            segments: Number of rotation segments
            axis: Axis to revolve around
            
        Returns:
            3D revolved voxel array
        """
        profile = voxels_2d
        height = profile.shape[1] if axis == 'y' else profile.shape[0]
        max_radius = profile.shape[0]
        
        dim = max_radius * 2 + 2
        result = np.zeros((dim, height, dim), dtype=np.uint8)
        center = dim // 2
        
        for angle_idx in range(segments):
            angle = 2 * math.pi * angle_idx / segments
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            
            for r in range(max_radius):
                for h in range(height):
                    if profile[r, h] > 0:
                        x = int(center + r * cos_a)
                        z = int(center + r * sin_a)
                        
                        if 0 <= x < dim and 0 <= z < dim:
                            result[x, h, z] = profile[r, h]
        
        return result
    
    @staticmethod
    def heightmap_to_voxels(heightmap: np.ndarray, max_height: int = 64,
                            value: int = 1, fill: bool = True) -> np.ndarray:
        """
        Convert a 2D heightmap to 3D voxels.
        
        Args:
            heightmap: 2D numpy array of height values (0-1 normalized)
            max_height: Maximum height in voxels
            value: Voxel value to use
            fill: If True, fill solid; if False, only surface
            
        Returns:
            3D voxel array
        """
        width, depth = heightmap.shape
        result = np.zeros((width, max_height, depth), dtype=np.uint8)
        
        for x in range(width):
            for z in range(depth):
                height = int(heightmap[x, z] * (max_height - 1))
                
                if fill:
                    result[x, :height + 1, z] = value
                else:
                    if height >= 0:
                        result[x, height, z] = value
        
        return result
    
    @staticmethod
    def create_text(text: str, font_size: int = 8, 
                    depth: int = 1, value: int = 1) -> np.ndarray:
        """
        Create 3D voxel text.
        
        Args:
            text: Text string to render
            font_size: Font size in pixels
            depth: Extrusion depth
            value: Voxel value
            
        Returns:
            3D voxel array with text
        """
        from PIL import Image, ImageDraw, ImageFont
        
        # Create image for text
        img = Image.new('L', (len(text) * font_size, font_size * 2), color=0)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        draw.text((0, 0), text, font=font, fill=255)
        
        # Convert to numpy
        text_array = np.array(img)
        
        # Find bounding box
        non_zero = np.argwhere(text_array > 128)
        if len(non_zero) == 0:
            return np.zeros((1, 1, 1), dtype=np.uint8)
        
        min_y, min_x = non_zero.min(axis=0)
        max_y, max_x = non_zero.max(axis=0)
        
        cropped = text_array[min_y:max_y + 1, min_x:max_x + 1]
        
        # Create 3D array
        result = np.zeros((cropped.shape[1], cropped.shape[0], depth), dtype=np.uint8)
        
        for y in range(cropped.shape[0]):
            for x in range(cropped.shape[1]):
                if cropped[y, x] > 128:
                    for z in range(depth):
                        result[x, cropped.shape[0] - 1 - y, z] = value
        
        return result
