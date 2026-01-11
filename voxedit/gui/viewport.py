"""
VoxelViewport - OpenGL 3D Viewport
==================================

Hardware-accelerated 3D viewport for voxel model visualization
using OpenGL for rendering and PyQt6 for integration.
"""

import numpy as np
from typing import Tuple, Optional, List
import math

from PyQt6.QtWidgets import QWidget
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QKeyEvent

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False
    print("Warning: PyOpenGL not found. 3D viewport will be disabled.")

from voxedit.core.voxel_model import VoxelModel
from voxedit.core.palette import VoxelPalette


class Camera:
    """
    Orbital camera for 3D viewport.
    
    Provides smooth orbiting, panning, and zooming controls.
    """
    
    def __init__(self):
        self.target = [0.0, 0.0, 0.0]  # Look-at point
        self.distance = 50.0  # Distance from target
        self.yaw = 45.0  # Horizontal rotation (degrees)
        self.pitch = 30.0  # Vertical rotation (degrees)
        
        # Limits
        self.min_distance = 5.0
        self.max_distance = 500.0
        self.min_pitch = -89.0
        self.max_pitch = 89.0
        
        # Smoothing
        self.smooth_factor = 0.15
        self._target_yaw = self.yaw
        self._target_pitch = self.pitch
        self._target_distance = self.distance
        self._target_target = list(self.target)
    
    @property
    def position(self) -> Tuple[float, float, float]:
        """Calculate camera position from orbital parameters."""
        pitch_rad = math.radians(self.pitch)
        yaw_rad = math.radians(self.yaw)
        
        x = self.target[0] + self.distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        y = self.target[1] + self.distance * math.sin(pitch_rad)
        z = self.target[2] + self.distance * math.cos(pitch_rad) * math.cos(yaw_rad)
        
        return (x, y, z)
    
    def orbit(self, delta_yaw: float, delta_pitch: float):
        """Rotate the camera around the target."""
        self._target_yaw += delta_yaw
        self._target_pitch = max(self.min_pitch, min(self.max_pitch, 
                                                      self._target_pitch + delta_pitch))
    
    def pan(self, delta_x: float, delta_y: float):
        """Pan the camera (move target)."""
        yaw_rad = math.radians(self.yaw)
        
        # Calculate right and up vectors
        right_x = math.cos(yaw_rad)
        right_z = -math.sin(yaw_rad)
        
        factor = self.distance * 0.002
        
        self._target_target[0] -= delta_x * right_x * factor
        self._target_target[2] -= delta_x * right_z * factor
        self._target_target[1] += delta_y * factor
    
    def zoom(self, delta: float):
        """Zoom in or out."""
        factor = 1.0 + delta * 0.001
        self._target_distance = max(self.min_distance, 
                                     min(self.max_distance, 
                                         self._target_distance * factor))
    
    def update(self):
        """Apply smoothing to camera movement."""
        self.yaw += (self._target_yaw - self.yaw) * self.smooth_factor
        self.pitch += (self._target_pitch - self.pitch) * self.smooth_factor
        self.distance += (self._target_distance - self.distance) * self.smooth_factor
        
        for i in range(3):
            self.target[i] += (self._target_target[i] - self.target[i]) * self.smooth_factor
    
    def focus_on(self, center: Tuple[float, float, float], size: float):
        """Focus camera on a point with appropriate distance."""
        self._target_target = list(center)
        self._target_distance = size * 2.0
    
    def reset(self):
        """Reset camera to default position."""
        self._target_yaw = 45.0
        self._target_pitch = 30.0
        self._target_distance = 50.0
        self._target_target = [0.0, 0.0, 0.0]


class VoxelRenderer:
    """
    OpenGL renderer for voxel models.
    
    Uses display lists or VBOs for efficient rendering.
    """
    
    def __init__(self):
        self.display_list = None
        self.voxel_count = 0
        self.show_grid = True
        self.show_axes = True
        self.show_wireframe = False
        self.ambient_occlusion = True
        self.model_size = (0, 0, 0)
    
    def build_mesh(self, model: VoxelModel, palette: VoxelPalette):
        """Build OpenGL geometry from voxel model."""
        if not HAS_OPENGL:
            return
        
        # Delete old display list
        if self.display_list is not None:
            glDeleteLists(self.display_list, 1)
        
        self.display_list = glGenLists(1)
        self.model_size = model.size
        self.voxel_count = model.get_voxel_count()
        
        glNewList(self.display_list, GL_COMPILE)
        
        # Get surface voxels for efficient rendering
        surface_voxels = model.get_surface_voxels()
        
        # Center offset
        offset = (-model.size[0] / 2, -model.size[1] / 2, -model.size[2] / 2)
        
        glBegin(GL_QUADS)
        
        for x, y, z, color_idx in surface_voxels:
            color = palette.get_color(color_idx)
            
            # Render visible faces
            self._render_voxel_faces(model, x, y, z, color, offset)
        
        glEnd()
        
        glEndList()
    
    def _render_voxel_faces(self, model: VoxelModel, x: int, y: int, z: int,
                            color, offset: Tuple[float, float, float]):
        """Render visible faces of a single voxel."""
        px = x + offset[0]
        py = y + offset[1]
        pz = z + offset[2]
        
        r, g, b = color.r / 255.0, color.g / 255.0, color.b / 255.0
        
        # Check each face
        # Right (+X)
        if not model.is_valid_position(x + 1, y, z) or model.get_voxel(x + 1, y, z) == 0:
            ao = 0.9 if self.ambient_occlusion else 1.0
            glColor3f(r * ao, g * ao, b * ao)
            glNormal3f(1, 0, 0)
            glVertex3f(px + 1, py, pz)
            glVertex3f(px + 1, py + 1, pz)
            glVertex3f(px + 1, py + 1, pz + 1)
            glVertex3f(px + 1, py, pz + 1)
        
        # Left (-X)
        if not model.is_valid_position(x - 1, y, z) or model.get_voxel(x - 1, y, z) == 0:
            ao = 0.7 if self.ambient_occlusion else 1.0
            glColor3f(r * ao, g * ao, b * ao)
            glNormal3f(-1, 0, 0)
            glVertex3f(px, py, pz + 1)
            glVertex3f(px, py + 1, pz + 1)
            glVertex3f(px, py + 1, pz)
            glVertex3f(px, py, pz)
        
        # Top (+Y)
        if not model.is_valid_position(x, y + 1, z) or model.get_voxel(x, y + 1, z) == 0:
            ao = 1.0
            glColor3f(r * ao, g * ao, b * ao)
            glNormal3f(0, 1, 0)
            glVertex3f(px, py + 1, pz)
            glVertex3f(px, py + 1, pz + 1)
            glVertex3f(px + 1, py + 1, pz + 1)
            glVertex3f(px + 1, py + 1, pz)
        
        # Bottom (-Y)
        if not model.is_valid_position(x, y - 1, z) or model.get_voxel(x, y - 1, z) == 0:
            ao = 0.5 if self.ambient_occlusion else 1.0
            glColor3f(r * ao, g * ao, b * ao)
            glNormal3f(0, -1, 0)
            glVertex3f(px, py, pz + 1)
            glVertex3f(px, py, pz)
            glVertex3f(px + 1, py, pz)
            glVertex3f(px + 1, py, pz + 1)
        
        # Front (+Z)
        if not model.is_valid_position(x, y, z + 1) or model.get_voxel(x, y, z + 1) == 0:
            ao = 0.85 if self.ambient_occlusion else 1.0
            glColor3f(r * ao, g * ao, b * ao)
            glNormal3f(0, 0, 1)
            glVertex3f(px, py, pz + 1)
            glVertex3f(px + 1, py, pz + 1)
            glVertex3f(px + 1, py + 1, pz + 1)
            glVertex3f(px, py + 1, pz + 1)
        
        # Back (-Z)
        if not model.is_valid_position(x, y, z - 1) or model.get_voxel(x, y, z - 1) == 0:
            ao = 0.75 if self.ambient_occlusion else 1.0
            glColor3f(r * ao, g * ao, b * ao)
            glNormal3f(0, 0, -1)
            glVertex3f(px + 1, py, pz)
            glVertex3f(px, py, pz)
            glVertex3f(px, py + 1, pz)
            glVertex3f(px + 1, py + 1, pz)
    
    def render(self):
        """Render the voxel model."""
        if not HAS_OPENGL or self.display_list is None:
            return
        
        if self.show_wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glCallList(self.display_list)
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    
    def render_grid(self, size: int = 32):
        """Render a ground grid."""
        if not HAS_OPENGL:
            return
        
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        
        half = size // 2
        
        for i in range(-half, half + 1):
            # Grid color
            if i == 0:
                glColor3f(0.5, 0.5, 0.5)
            else:
                glColor3f(0.3, 0.3, 0.3)
            
            # X lines
            glVertex3f(i, 0, -half)
            glVertex3f(i, 0, half)
            
            # Z lines
            glVertex3f(-half, 0, i)
            glVertex3f(half, 0, i)
        
        glEnd()
        glEnable(GL_LIGHTING)
    
    def render_axes(self, size: float = 10.0):
        """Render coordinate axes."""
        if not HAS_OPENGL:
            return
        
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        
        glBegin(GL_LINES)
        
        # X axis (red)
        glColor3f(1.0, 0.2, 0.2)
        glVertex3f(0, 0, 0)
        glVertex3f(size, 0, 0)
        
        # Y axis (green)
        glColor3f(0.2, 1.0, 0.2)
        glVertex3f(0, 0, 0)
        glVertex3f(0, size, 0)
        
        # Z axis (blue)
        glColor3f(0.2, 0.2, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, size)
        
        glEnd()
        
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def render_bounds(self, size: Tuple[int, int, int]):
        """Render model bounding box."""
        if not HAS_OPENGL:
            return
        
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 0.0)
        
        x = size[0] / 2
        y = size[1] / 2
        z = size[2] / 2
        
        glBegin(GL_LINE_LOOP)
        glVertex3f(-x, -y, -z)
        glVertex3f(x, -y, -z)
        glVertex3f(x, -y, z)
        glVertex3f(-x, -y, z)
        glEnd()
        
        glBegin(GL_LINE_LOOP)
        glVertex3f(-x, y, -z)
        glVertex3f(x, y, -z)
        glVertex3f(x, y, z)
        glVertex3f(-x, y, z)
        glEnd()
        
        glBegin(GL_LINES)
        glVertex3f(-x, -y, -z)
        glVertex3f(-x, y, -z)
        glVertex3f(x, -y, -z)
        glVertex3f(x, y, -z)
        glVertex3f(x, -y, z)
        glVertex3f(x, y, z)
        glVertex3f(-x, -y, z)
        glVertex3f(-x, y, z)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def cleanup(self):
        """Clean up OpenGL resources."""
        if HAS_OPENGL and self.display_list is not None:
            glDeleteLists(self.display_list, 1)
            self.display_list = None


class VoxelViewport(QOpenGLWidget):
    """
    OpenGL viewport widget for displaying and editing voxel models.
    
    Signals:
        voxelClicked: Emitted when a voxel is clicked (x, y, z, button)
        voxelHovered: Emitted when mouse hovers over a voxel (x, y, z)
    """
    
    voxelClicked = pyqtSignal(int, int, int, int)  # x, y, z, button
    voxelHovered = pyqtSignal(int, int, int)  # x, y, z
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.camera = Camera()
        self.renderer = VoxelRenderer()
        
        self.model: Optional[VoxelModel] = None
        self.palette: Optional[VoxelPalette] = None
        
        # OpenGL context state
        self._gl_initialized = False
        self._needs_rebuild = False
        
        # Mouse state
        self._last_mouse_pos = QPoint()
        self._mouse_buttons = Qt.MouseButton.NoButton
        
        # Background color
        self.bg_color = (0.15, 0.15, 0.18)
        
        # Update timer for smooth camera
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._on_timer)
        self._update_timer.start(16)  # ~60 FPS
        
        # Enable mouse tracking for hover
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def set_model(self, model: VoxelModel, palette: VoxelPalette):
        """Set the voxel model to display."""
        self.model = model
        self.palette = palette
        
        if self._gl_initialized:
            self.rebuild_mesh()
        else:
            self._needs_rebuild = True
        
        # Focus camera on model
        center = (0, model.size[1] / 2, 0)
        size = max(model.size)
        self.camera.focus_on(center, size)
    
    def rebuild_mesh(self):
        """Rebuild the OpenGL mesh from the current model."""
        if not self._gl_initialized:
            self._needs_rebuild = True
            return
            
        if self.model is not None and self.palette is not None:
            self.renderer.build_mesh(self.model, self.palette)
            self.update()
    
    def initializeGL(self):
        """Initialize OpenGL state."""
        if not HAS_OPENGL:
            return
        
        glClearColor(*self.bg_color, 1.0)
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        
        # Enable face culling
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        # Enable smooth shading
        glShadeModel(GL_SMOOTH)
        
        # Set up lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Light properties
        light_pos = [50.0, 100.0, 50.0, 0.0]  # Directional light
        light_ambient = [0.3, 0.3, 0.35, 1.0]
        light_diffuse = [0.8, 0.8, 0.8, 1.0]
        light_specular = [0.3, 0.3, 0.3, 1.0]
        
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
        
        # Enable alpha blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Mark OpenGL as initialized
        self._gl_initialized = True
        
        # Build mesh if it was deferred
        if self._needs_rebuild:
            self._needs_rebuild = False
            self.rebuild_mesh()
    
    def resizeGL(self, width: int, height: int):
        """Handle viewport resize."""
        if not HAS_OPENGL:
            return
        
        glViewport(0, 0, width, height)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect = width / height if height > 0 else 1.0
        gluPerspective(45.0, aspect, 0.1, 1000.0)
        
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        """Render the scene."""
        if not HAS_OPENGL:
            return
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Set up camera
        pos = self.camera.position
        target = self.camera.target
        
        gluLookAt(
            pos[0], pos[1], pos[2],
            target[0], target[1], target[2],
            0.0, 1.0, 0.0
        )
        
        # Update light position
        light_pos = [50.0, 100.0, 50.0, 0.0]
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
        
        # Render grid
        if self.renderer.show_grid:
            self.renderer.render_grid(64)
        
        # Render axes
        if self.renderer.show_axes:
            self.renderer.render_axes(10.0)
        
        # Render model
        self.renderer.render()
        
        # Render bounds
        if self.model is not None:
            self.renderer.render_bounds(self.model.size)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        self._last_mouse_pos = event.pos()
        self._mouse_buttons = event.buttons()
        
        # Handle voxel clicking
        if event.button() == Qt.MouseButton.LeftButton:
            # TODO: Implement ray casting for voxel selection
            pass
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement."""
        delta = event.pos() - self._last_mouse_pos
        self._last_mouse_pos = event.pos()
        
        if event.buttons() & Qt.MouseButton.RightButton:
            # Orbit camera
            self.camera.orbit(-delta.x() * 0.3, -delta.y() * 0.3)
            self.update()
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            # Pan camera
            self.camera.pan(delta.x(), delta.y())
            self.update()
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel."""
        delta = event.angleDelta().y()
        self.camera.zoom(-delta)
        self.update()
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press."""
        key = event.key()
        
        if key == Qt.Key.Key_Home:
            self.camera.reset()
        elif key == Qt.Key.Key_G:
            self.renderer.show_grid = not self.renderer.show_grid
        elif key == Qt.Key.Key_A:
            self.renderer.show_axes = not self.renderer.show_axes
        elif key == Qt.Key.Key_W:
            self.renderer.show_wireframe = not self.renderer.show_wireframe
        
        self.update()
    
    def _on_timer(self):
        """Timer callback for smooth camera updates."""
        self.camera.update()
        self.update()
    
    def set_background_color(self, r: float, g: float, b: float):
        """Set the viewport background color."""
        self.bg_color = (r, g, b)
        if HAS_OPENGL:
            self.makeCurrent()
            glClearColor(r, g, b, 1.0)
            self.update()
    
    def screenshot(self) -> Optional[np.ndarray]:
        """Capture a screenshot of the viewport."""
        if not HAS_OPENGL:
            return None
        
        self.makeCurrent()
        width, height = self.width(), self.height()
        
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
        
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)
        image = np.flipud(image)  # Flip vertically
        
        return image
    
    def closeEvent(self, event):
        """Clean up on close."""
        self._update_timer.stop()
        self.renderer.cleanup()
        super().closeEvent(event)
