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
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint, QPointF
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QKeyEvent, QPainter, QColor, QFont

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False
    print("Warning: PyOpenGL not found. 3D viewport will be disabled.")

from voxedit.core.voxel_model import VoxelModel
from voxedit.core.palette import VoxelPalette
from voxedit.gui.tool_panel import Tool


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
        
        # Compute render-space center offset (respect swap)
        rsx, rsy, rsz = (model.size[0], model.size[2], model.size[1]) if getattr(self, 'swap_yz', False) else model.size
        offset = (-rsx / 2, -rsy / 2, -rsz / 2)
        
        glBegin(GL_QUADS)
        
        for x, y, z, color_idx in surface_voxels:
            color = palette.get_color(color_idx)

            # Map model voxel to render/world coordinates if axes are swapped
            if getattr(self, 'swap_yz', False):
                rx, ry, rz = x, z, y
            else:
                rx, ry, rz = x, y, z
            
            # Render visible faces using render coords
            self._render_voxel_faces(model, rx, ry, rz, color, offset)
        
        glEnd()
        
        glEndList()
    
    def _render_voxel_faces(self, model: VoxelModel, x: int, y: int, z: int,
                            color, offset: Tuple[float, float, float]):
        """Render visible faces of a single voxel.
        Note: the incoming (x,y,z) here are in *render/world* coordinates and may have
        been remapped from model coords if Y/Z swapping is enabled."""
        px = x + offset[0]
        py = y + offset[1]
        pz = z + offset[2]
        
        r, g, b = color.r / 255.0, color.g / 255.0, color.b / 255.0
        
        # When checking neighbors for face visibility and fetching voxel values we must
        # map back to model-space coordinates if we swapped axes earlier.
        def model_neighbor(nx, ny, nz):
            if getattr(self, 'swap_yz', False):
                # incoming args are in render coords; map back: (rx,ry,rz) -> (x, z, y)
                mx, my, mz = nx, nz, ny
            else:
                mx, my, mz = nx, ny, nz
            return mx, my, mz
        
        # Check each face
        # Right (+X)
        mx, my, mz = model_neighbor(x + 1, y, z)
        if not model.is_valid_position(mx, my, mz) or model.get_voxel(mx, my, mz) == 0:
            ao = 0.9 if self.ambient_occlusion else 1.0
            glColor3f(r * ao, g * ao, b * ao)
            glNormal3f(1, 0, 0)
            glVertex3f(px + 1, py, pz)
            glVertex3f(px + 1, py + 1, pz)
            glVertex3f(px + 1, py + 1, pz + 1)
            glVertex3f(px + 1, py, pz + 1)
        
        # Left (-X)
        mx, my, mz = model_neighbor(x - 1, y, z)
        if not model.is_valid_position(mx, my, mz) or model.get_voxel(mx, my, mz) == 0:
            ao = 0.7 if self.ambient_occlusion else 1.0
            glColor3f(r * ao, g * ao, b * ao)
            glNormal3f(-1, 0, 0)
            glVertex3f(px, py, pz + 1)
            glVertex3f(px, py + 1, pz + 1)
            glVertex3f(px, py + 1, pz)
            glVertex3f(px, py, pz)
        

        
        # Top (+Y)
        mx, my, mz = model_neighbor(x, y + 1, z)
        if not model.is_valid_position(mx, my, mz) or model.get_voxel(mx, my, mz) == 0:
            ao = 1.0
            glColor3f(r * ao, g * ao, b * ao)
            glNormal3f(0, 1, 0)
            glVertex3f(px, py + 1, pz)
            glVertex3f(px, py + 1, pz + 1)
            glVertex3f(px + 1, py + 1, pz + 1)
            glVertex3f(px + 1, py + 1, pz)
        
        # Bottom (-Y)
        mx, my, mz = model_neighbor(x, y - 1, z)
        if not model.is_valid_position(mx, my, mz) or model.get_voxel(mx, my, mz) == 0:
            ao = 0.5 if self.ambient_occlusion else 1.0
            glColor3f(r * ao, g * ao, b * ao)
            glNormal3f(0, -1, 0)
            glVertex3f(px, py, pz + 1)
            glVertex3f(px, py, pz)
            glVertex3f(px + 1, py, pz)
            glVertex3f(px + 1, py, pz + 1)
        
        # Front (+Z)
        mx, my, mz = model_neighbor(x, y, z + 1)
        if not model.is_valid_position(mx, my, mz) or model.get_voxel(mx, my, mz) == 0:
            ao = 0.85 if self.ambient_occlusion else 1.0
            glColor3f(r * ao, g * ao, b * ao)
            glNormal3f(0, 0, 1)
            glVertex3f(px, py, pz + 1)
            glVertex3f(px + 1, py, pz + 1)
            glVertex3f(px + 1, py + 1, pz + 1)
            glVertex3f(px, py + 1, pz + 1)
        
        # Back (-Z)
        mx, my, mz = model_neighbor(x, y, z - 1)
        if not model.is_valid_position(mx, my, mz) or model.get_voxel(mx, my, mz) == 0:
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
    cursorMoved = pyqtSignal(int, int, int)  # x, y, z
    toolDragStarted = pyqtSignal()
    toolDragEnded = pyqtSignal()
    axisLockChanged = pyqtSignal(object)  # axis char 'x'/'y'/'z' or None
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.camera = Camera()
        self.renderer = VoxelRenderer()
        
        self.model: Optional[VoxelModel] = None
        self.palette: Optional[VoxelPalette] = None
        
        # Option to swap model Y/Z axes for rendering/input mapping if needed
        # Set to True if your model data uses (x, z, y) ordering and you want Y to be vertical
        self.swap_yz = False

        # 3D cursor state
        self.cursor_position: Optional[Tuple[int, int, int]] = None
        self.show_cursor = True
        
        # OpenGL context state
        self._gl_initialized = False
        self._needs_rebuild = False
        
        # Mouse state (use floating point position to support HiDPI fractional coords)
        self._last_mouse_pos = QPointF(0.0, 0.0)
        self._mouse_buttons = Qt.MouseButton.NoButton
        # Drag tracking (detect continuous tool dragging separate from single click)
        self._drag_active = False
        # Remember last click position to avoid duplicate emission on tiny initial move
        self._last_click_pos = None
        # Axis locks: allow multiple axes to be locked simultaneously
        # _axis_locks stores a set of axes currently locked (e.g., {'x', 'y'})
        # _axis_lock_anchors maps axis -> anchor coordinate tuple (x,y,z)
        self._axis_locks = set()
        self._axis_lock_anchors: dict[str, tuple[int, int, int]] = {}
        # Track currently pressed non-modifier keys to ignore autorepeat noise
        self._pressed_keys = set()
        
        # Background color
        self.bg_color = (0.15, 0.15, 0.18)
        
        # Update timer for smooth camera
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._on_timer)
        self._update_timer.start(16)  # ~60 FPS
        
        # Enable mouse tracking for hover
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def _model_to_world(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Map model coordinates to world coordinates, applying Y<->Z swap if enabled."""
        if not self.swap_yz:
            return (x, y, z)
        return (x, z, y)

    def _world_to_model(self, pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Inverse of _model_to_world."""
        if not self.swap_yz:
            return pos
        return (pos[0], pos[2], pos[1])

    def _get_render_size(self) -> Tuple[int, int, int]:
        """Return the model size in world/render order (apply swap if necessary)."""
        sx, sy, sz = self.model.size
        return (sx, sz, sy) if self.swap_yz else (sx, sy, sz)

    def set_model(self, model: VoxelModel, palette: VoxelPalette):
        """Set the voxel model to display."""
        self.model = model
        self.palette = palette
        
        if self._gl_initialized:
            self.rebuild_mesh()
        else:
            self._needs_rebuild = True
        
        # Focus camera on model (convert model center to world coords)
        model_center = (0, model.size[1] / 2, 0)
        world_center = self._model_to_world(*model_center)
        size = max(model.size)
        self.camera.focus_on(world_center, size)
    
    def rebuild_mesh(self):
        """Rebuild the OpenGL mesh from the current model."""
        if not self._gl_initialized:
            self._needs_rebuild = True
            return
            
        if self.model is not None and self.palette is not None:
            # Propagate swap flag to renderer so rendering order matches world mapping
            self.renderer.swap_yz = self.swap_yz
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
        """Handle viewport resize — account for HiDPI device pixel ratio."""
        if not HAS_OPENGL:
            return

        # Account for HiDPI scaling: Qt passes device-independent size, but OpenGL viewport
        # should be set in device pixels.
        try:
            scale = float(self.devicePixelRatioF())
        except Exception:
            try:
                scale = float(self.devicePixelRatio())
            except Exception:
                scale = 1.0

        vp_width = max(1, int(round(width * scale)))
        vp_height = max(1, int(round(height * scale)))

        glViewport(0, 0, vp_width, vp_height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # Aspect ratio remains the same regardless of pixel scaling (scale cancels out)
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
        
        # Render bounds (in render/world coordinates)
        if self.model is not None:
            self.renderer.render_bounds(self._get_render_size())
        
        # Render 3D cursor
        if self.show_cursor and self.cursor_position is not None:
            self._render_cursor()

        # Render overlay (axis lock status, anchor)
        self._render_overlay()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        # Ensure we have keyboard focus when clicked so key events (X/Y/Z) will be received while interacting
        self.setFocus()

        # Use floating position for HiDPI accuracy
        try:
            pos = event.position()
        except Exception:
            pos = event.pos()
        self._last_mouse_pos = pos
        self._mouse_buttons = event.buttons()

        # Handle voxel clicking
        if event.button() == Qt.MouseButton.LeftButton:
            # Resolve surface and placement coordinates
            surface, placement = self._get_surface_and_placement_at_mouse(pos)

            # Try to determine current tool from parent if available
            tool = None
            parent = self.parent()
            if parent is not None and hasattr(parent, 'tool_panel'):
                tool = parent.tool_panel.get_current_tool()

            chosen = None
            # Pencil/shape tools prefer placement (empty voxel) when available
            if tool in (Tool.PENCIL, Tool.BOX, Tool.SPHERE, Tool.LINE):
                chosen = placement if placement is not None else surface
            # Eraser/paint/select/eyedropper/fill prefer the surface voxel
            elif tool in (Tool.ERASER, Tool.PAINT, Tool.SELECT, Tool.EYEDROPPER, Tool.FILL):
                chosen = surface if surface is not None else placement
            else:
                # Fallback: surface then placement
                chosen = surface if surface is not None else placement

            if chosen is not None:
                x, y, z = chosen

                # Apply axis locks to the chosen voxel if active (supports multiple locks)
                if self._axis_locks:
                    for ax in list(self._axis_locks):
                        anchor = self._axis_lock_anchors.get(ax)
                        if anchor is None:
                            continue
                        if ax == 'x':
                            y, z = anchor[1], anchor[2]
                        elif ax == 'y':
                            x, z = anchor[0], anchor[2]
                        elif ax == 'z':
                            x, y = anchor[0], anchor[1]

                    chosen = (x, y, z)

                # Record that we are not yet dragging (movement will trigger drag start)
                self._drag_active = False
                # Emit initial click immediately for single-click tools
                self.voxelClicked.emit(x, y, z, event.button())
                # Remember the pressed voxel to avoid emitting it again on the first tiny move
                self._last_click_pos = (x, y, z)
                # Remember the pressed voxel to avoid emitting it again on the first tiny move
                self._last_click_pos = (x, y, z)
    
    def _get_voxel_at_mouse(self, mouse_pos: QPoint) -> Optional[Tuple[int, int, int]]:
        """Cast ray from mouse position and find intersected voxel."""
        if self.model is None:
            return None
        
        # Use window coordinates (pixels) and convert to OpenGL bottom-left origin
        width, height = self.width(), self.height()
        if width == 0 or height == 0:
            return None

        # Account for device pixel ratio (HiDPI displays) — OpenGL viewport uses device pixels
        scale = 1.0
        try:
            scale = float(self.devicePixelRatioF())
        except Exception:
            # Fall back on integer DPI if function missing
            try:
                scale = float(self.devicePixelRatio())
            except Exception:
                scale = 1.0

        win_x = float(mouse_pos.x()) * scale
        win_y = float(height) * scale - (float(mouse_pos.y()) * scale)  # OpenGL origin is bottom-left

        # Create ray from camera through mouse window position
        ray_origin, ray_direction = self._screen_to_world_ray(win_x, win_y)
        
        # Find intersection with voxels (backwards-compatible single coordinate)
        result = self._ray_voxel_intersection(ray_origin, ray_direction)
        return result

    def _get_surface_and_placement_at_mouse(self, mouse_pos: QPoint) -> Tuple[Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int]]]:
        """Return (surface_voxel, placement_voxel) at mouse position.

        surface_voxel: first occupied voxel hit along ray (closest to camera)
        placement_voxel: empty voxel just in front of the surface (where a new voxel would be placed),
                         or the voxel at bbox intersection if no occupied voxel present.
        """
        if self.model is None:
            return None, None

        # Compute window coords (scale-aware)
        width, height = self.width(), self.height()
        if width == 0 or height == 0:
            return None, None

        scale = 1.0
        try:
            scale = float(self.devicePixelRatioF())
        except Exception:
            try:
                scale = float(self.devicePixelRatio())
            except Exception:
                scale = 1.0

        # Convert widget coords to device (GL) pixel coords using viewport height
        win_x = float(mouse_pos.x()) * scale
        win_y = float(height) * scale - (float(mouse_pos.y()) * scale)

        ray_origin, ray_direction = self._screen_to_world_ray(win_x, win_y)
        return self._ray_voxel_hit(ray_origin, ray_direction)

    def _ray_voxel_hit(self, ray_origin: Tuple[float, float, float], ray_direction: Tuple[float, float, float]) -> Tuple[Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int]]]:
        """Perform ray tracing through the model and return (surface_voxel, placement_voxel).

        surface_voxel: coordinates of the first non-empty voxel hit (closest to camera)
        placement_voxel: the empty voxel immediately before the surface (towards the camera),
                         or the bbox intersection voxel if no surface is hit.
        """
        if self.model is None:
            return None, None

        # Normalize ray direction
        dir_length = math.sqrt(ray_direction[0]**2 + ray_direction[1]**2 + ray_direction[2]**2)
        if dir_length == 0:
            return None, None

        dir_x, dir_y, dir_z = (
            ray_direction[0] / dir_length,
            ray_direction[1] / dir_length,
            ray_direction[2] / dir_length
        )

        # Model bounds
        size_x, size_y, size_z = self.model.size
        half_x, half_y, half_z = size_x / 2, size_y / 2, size_z / 2

        # Build world-space bbox depending on swap
        if self.swap_yz:
            bbox_min = (-half_x, -half_z, -half_y)
            bbox_max = (half_x, half_z, half_y)
        else:
            bbox_min = (-half_x, -half_y, -half_z)
            bbox_max = (half_x, half_y, half_z)

        bbox_intersection = self._ray_bbox_intersection(ray_origin, ray_direction,
                                                       bbox_min,
                                                       bbox_max)
        if bbox_intersection is None:
            return None, None

        start_t = bbox_intersection
        max_distance = 200.0
        step_size = 0.1

        last_voxel = None
        for t in np.arange(start_t, start_t + max_distance, step_size):
            current_pos = (
                ray_origin[0] + t * dir_x,
                ray_origin[1] + t * dir_y,
                ray_origin[2] + t * dir_z
            )

            # Map world position to model coords (invert swap if needed)
            mp = self._world_to_model(current_pos)

            vx = int(round(mp[0] + half_x))
            vy = int(round(mp[1] + half_y))
            vz = int(round(mp[2] + half_z))

            if last_voxel == (vx, vy, vz):
                continue
            last_voxel = (vx, vy, vz)

            if not (0 <= vx < size_x and 0 <= vy < size_y and 0 <= vz < size_z):
                continue

            # Read voxel value at this position
            val = self.model.get_voxel(vx, vy, vz)

            # If it's empty, this is a valid placement voxel (first empty along ray)
            if val == 0:
                return None, (vx, vy, vz)

            if val > 0:
                # Surface voxel found
                surface = (vx, vy, vz)

                # Compute placement voxel one step back along the ray
                back_t = max(start_t, t - step_size)
                back_pos = (
                    ray_origin[0] + back_t * dir_x,
                    ray_origin[1] + back_t * dir_y,
                    ray_origin[2] + back_t * dir_z
                )
                bmp = self._world_to_model(back_pos)
                bx = int(round(bmp[0] + half_x))
                by = int(round(bmp[1] + half_y))
                bz = int(round(bmp[2] + half_z))

                if (0 <= bx < size_x and 0 <= by < size_y and 0 <= bz < size_z):
                    if self.model.get_voxel(bx, by, bz) == 0:
                        placement = (bx, by, bz)
                        return surface, placement
                # If cannot place in front, return surface and None for placement
                return surface, None

        # No surface found; return bbox intersection voxel as placement
        intersection_pos = (
            ray_origin[0] + start_t * dir_x,
            ray_origin[1] + start_t * dir_y,
            ray_origin[2] + start_t * dir_z
        )

        ip = self._world_to_model(intersection_pos)
        vx = int(round(ip[0] + half_x))
        vy = int(round(ip[1] + half_y))
        vz = int(round(ip[2] + half_z))

        if (0 <= vx < size_x and 0 <= vy < size_y and 0 <= vz < size_z):
            return None, (vx, vy, vz)

        return None, None
    
    def _screen_to_world_ray(self, win_x: float, win_y: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Convert window (pixel) coordinates to a world space ray.

        Uses the current camera transform to build a modelview matrix before
        calling gluUnProject so that the unprojected points match the view
        used for rendering.
        """
        # Ensure OpenGL context is current and set the camera matrix
        if HAS_OPENGL:
            self.makeCurrent()
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            pos = self.camera.position
            target = self.camera.target
            gluLookAt(
                pos[0], pos[1], pos[2],
                target[0], target[1], target[2],
                0.0, 1.0, 0.0
            )

        # Read projection and modelview matrices and viewport
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)

        # Unproject near and far points using window coordinates
        near_pos = gluUnProject(win_x, win_y, 0.0, modelview, projection, viewport)
        far_pos = gluUnProject(win_x, win_y, 1.0, modelview, projection, viewport)

        if HAS_OPENGL:
            glPopMatrix()

        # Use the unprojected near-plane point as the ray origin to avoid camera-position bias
        ray_origin = (near_pos[0], near_pos[1], near_pos[2])
        ray_direction = (
            far_pos[0] - near_pos[0],
            far_pos[1] - near_pos[1],
            far_pos[2] - near_pos[2]
        )

        # Normalize ray direction
        length = math.sqrt(sum(d * d for d in ray_direction))
        if length > 0:
            ray_direction = tuple(d / length for d in ray_direction)

        return ray_origin, ray_direction
    
    def _ray_voxel_intersection(self, ray_origin: Tuple[float, float, float], 
                               ray_direction: Tuple[float, float, float]) -> Optional[Tuple[int, int, int]]:
        """Find the voxel position where the ray intersects with existing voxels or model surface."""
        if self.model is None:
            return None
        
        # Normalize ray direction
        dir_length = math.sqrt(ray_direction[0]**2 + ray_direction[1]**2 + ray_direction[2]**2)
        if dir_length == 0:
            return None
        
        dir_x, dir_y, dir_z = (
            ray_direction[0] / dir_length,
            ray_direction[1] / dir_length,
            ray_direction[2] / dir_length
        )
        
        # Get model bounds (centered at origin in world space)
        size_x, size_y, size_z = self.model.size
        half_x, half_y, half_z = size_x / 2, size_y / 2, size_z / 2
        
        # First, find intersection with model bounding box (in world coordinates)
        if self.swap_yz:
            bbox_min = (-half_x, -half_z, -half_y)
            bbox_max = (half_x, half_z, half_y)
        else:
            bbox_min = (-half_x, -half_y, -half_z)
            bbox_max = (half_x, half_y, half_z)
        bbox_intersection = self._ray_bbox_intersection(ray_origin, ray_direction, bbox_min, bbox_max)
        
        if bbox_intersection is None:
            return None
        
        # Start from the bounding box intersection point
        start_t = bbox_intersection
        max_distance = 200.0
        step_size = 0.1  # Smaller step size for better precision
        
        # Step through the ray and find the first occupied voxel
        prev_pos = None
        for t in np.arange(start_t, start_t + max_distance, step_size):
            # Calculate current position along ray
            current_pos = (
                ray_origin[0] + t * dir_x,
                ray_origin[1] + t * dir_y,
                ray_origin[2] + t * dir_z
            )
            
            # Map world position to model coordinates and convert to voxel indices
            mp = self._world_to_model(current_pos)
            vx = int(round(mp[0] + half_x))
            vy = int(round(mp[1] + half_y))
            vz = int(round(mp[2] + half_z))
            
            # Save previous position (closest to camera) for potential placement
            prev_pos = (vx, vy, vz)
            
            # Check if within model bounds
            if (0 <= vx < size_x and 0 <= vy < size_y and 0 <= vz < size_z):
                # Check if this voxel is occupied
                if self.model.get_voxel(vx, vy, vz) > 0:
                    # Found an occupied voxel. For placement, prefer the voxel just in front
                    # of this surface (i.e., the voxel that is closer to the camera).
                    # Compute previous position along the ray (one step back)
                    back_pos = (
                        ray_origin[0] + (t - step_size) * dir_x,
                        ray_origin[1] + (t - step_size) * dir_y,
                        ray_origin[2] + (t - step_size) * dir_z
                    )
                    bmp = self._world_to_model(back_pos)
                    bx = int(round(bmp[0] + half_x))
                    by = int(round(bmp[1] + half_y))
                    bz = int(round(bmp[2] + half_z))

                    # If the back position is a valid empty voxel, return it for placement
                    if (0 <= bx < size_x and 0 <= by < size_y and 0 <= bz < size_z):
                        if self.model.get_voxel(bx, by, bz) == 0:
                            return (bx, by, bz)

                    # Otherwise fall back to returning the surface voxel itself
        
        # If no occupied voxel found, return the position at the bounding box intersection
        # This ensures cursor shows even when pointing at empty space within model bounds
        intersection_pos = (
            ray_origin[0] + start_t * dir_x,
            ray_origin[1] + start_t * dir_y,
            ray_origin[2] + start_t * dir_z
        )
        
        ip = self._world_to_model(intersection_pos)
        vx = int(round(ip[0] + half_x))
        vy = int(round(ip[1] + half_y))
        vz = int(round(ip[2] + half_z))
        
        if (0 <= vx < size_x and 0 <= vy < size_y and 0 <= vz < size_z):
            return (vx, vy, vz)
        
        return None
    
    def _ray_bbox_intersection(self, ray_origin: Tuple[float, float, float], 
                              ray_direction: Tuple[float, float, float],
                              bbox_min: Tuple[float, float, float],
                              bbox_max: Tuple[float, float, float]) -> Optional[float]:
        """Find the distance to the first intersection with an axis-aligned bounding box."""
        # Normalize ray direction
        dir_length = math.sqrt(ray_direction[0]**2 + ray_direction[1]**2 + ray_direction[2]**2)
        if dir_length == 0:
            return None
        
        dir_x, dir_y, dir_z = (
            ray_direction[0] / dir_length,
            ray_direction[1] / dir_length,
            ray_direction[2] / dir_length
        )
        
        # Calculate intersection with bounding box using slab method
        t_min = 0.0
        t_max = float('inf')
        
        # X slab
        if abs(dir_x) > 1e-6:
            t1 = (bbox_min[0] - ray_origin[0]) / dir_x
            t2 = (bbox_max[0] - ray_origin[0]) / dir_x
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
        else:
            # Ray is parallel to X planes
            if ray_origin[0] < bbox_min[0] or ray_origin[0] > bbox_max[0]:
                return None
        
        # Y slab
        if abs(dir_y) > 1e-6:
            t1 = (bbox_min[1] - ray_origin[1]) / dir_y
            t2 = (bbox_max[1] - ray_origin[1]) / dir_y
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
        else:
            # Ray is parallel to Y planes
            if ray_origin[1] < bbox_min[1] or ray_origin[1] > bbox_max[1]:
                return None
        
        # Z slab
        if abs(dir_z) > 1e-6:
            t1 = (bbox_min[2] - ray_origin[2]) / dir_z
            t2 = (bbox_max[2] - ray_origin[2]) / dir_z
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
        else:
            # Ray is parallel to Z planes
            if ray_origin[2] < bbox_min[2] or ray_origin[2] > bbox_max[2]:
                return None
        
        if t_min > t_max or t_max < 0:
            return None
        
        # Return the first intersection point (closest to ray origin)
        return max(0.0, t_min)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement."""
        # Ensure viewport has keyboard focus when the mouse is moving over it so key events (X/Y/Z) are received
        self.setFocus()

        try:
            pos = event.position()
        except Exception:
            pos = event.pos()
        delta = pos - self._last_mouse_pos
        self._last_mouse_pos = pos
        
        # Update 3D cursor position
        if self.show_cursor:
            surface, placement = self._get_surface_and_placement_at_mouse(event.pos())
            # Prefer showing the placement (empty voxel) when available so cursor can enter empty spaces; fallback to surface
            new_cursor = placement if placement is not None else surface

            # If axis locks are active, constrain the cursor to the locked axes using anchors
            if new_cursor is not None and self._axis_locks:
                nx, ny, nz = new_cursor
                for ax in list(self._axis_locks):
                    anchor = self._axis_lock_anchors.get(ax)
                    if anchor is None:
                        continue
                    if ax == 'x':
                        nx, ny, nz = nx, anchor[1], anchor[2]
                    elif ax == 'y':
                        nx, ny, nz = anchor[0], ny, anchor[2]
                    elif ax == 'z':
                        nx, ny, nz = anchor[0], anchor[1], nz
                new_cursor = (nx, ny, nz)

            if new_cursor != self.cursor_position:
                self.cursor_position = new_cursor
                self.update()  # Redraw to show cursor
                if self.cursor_position is not None:
                    x, y, z = self.cursor_position
                    self.voxelHovered.emit(x, y, z)  # Emit hover signal
                else:
                    # Emit sentinel to indicate cleared hover
                    self.voxelHovered.emit(-1, -1, -1)
        
        if event.buttons() & Qt.MouseButton.RightButton:
            # Orbit camera
            self.camera.orbit(-delta.x() * 0.3, -delta.y() * 0.3)
            self.update()
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            # Pan camera
            self.camera.pan(delta.x(), delta.y())
            self.update()
        elif event.buttons() & Qt.MouseButton.LeftButton:
            # Apply tool continuously while dragging
            voxel_pos = self._get_voxel_at_mouse(event.pos())
            if voxel_pos is not None:
                x, y, z = voxel_pos
                # On first movement while left-button is down, start a drag operation
                if not getattr(self, '_drag_active', False):
                    self._drag_active = True
                    self.toolDragStarted.emit()
                    # If the first moved voxel matches the immediate press voxel, skip emitting
                    if self._last_click_pos == (x, y, z):
                        # Clear press memory and skip duplicate emission
                        self._last_click_pos = None
                    else:
                        self._last_click_pos = None
                        self.voxelClicked.emit(x, y, z, event.button())
                else:
                    self.voxelClicked.emit(x, y, z, event.button())

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release (end of dragging)."""
        # Emit drag ended so callers can finalize compound tools
        if event.button() == Qt.MouseButton.LeftButton:
            # If we were dragging, finalize the drag operation; otherwise this was a single click
            if getattr(self, '_drag_active', False):
                self.toolDragEnded.emit()
                self._drag_active = False
            # Do not emit an extra click on release (press already emitted one)
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel."""
        delta = event.angleDelta().y()
        self.camera.zoom(-delta)
        self.update()

    def set_axis_lock(self, axis: str):
        """Set axis lock for the given axis and compute its anchor (supports multiple locks)."""
        if axis not in ('x', 'y', 'z'):
            return
        if axis in self._axis_locks:
            return
        # Determine anchor mode from parent tool panel (cursor/surface/center)
        parent = self.parent()
        mode = 'cursor'
        if parent is not None and hasattr(parent, 'tool_panel'):
            try:
                mode = parent.tool_panel.get_axis_anchor_mode()
            except Exception:
                mode = 'cursor'

        anchor = None
        # Prefer cursor position when mode == 'cursor'
        if mode == 'cursor' and self.cursor_position is not None:
            anchor = self.cursor_position
        elif mode == 'surface':
            # Use last mouse position to pick surface/placement
            try:
                surface, placement = self._get_surface_and_placement_at_mouse(self._last_mouse_pos)
                anchor = surface if surface is not None else placement
            except Exception:
                anchor = None
        # Mode 'center' or fallback
        if anchor is None and self.model is not None:
            s = self.model.size
            anchor = (s[0]//2, s[1]//2, s[2]//2)

        # Store lock & anchor
        self._axis_locks.add(axis)
        self._axis_lock_anchors[axis] = anchor
        print(f"Axis lock set: {axis.upper()} at {anchor} (mode={mode})")

        # Immediately enforce the axis constraint on the current cursor position for the newly locked axis
        if self.cursor_position is not None:
            cx, cy, cz = self.cursor_position
            if axis == 'x':
                self.cursor_position = (cx, anchor[1], anchor[2])
            elif axis == 'y':
                self.cursor_position = (anchor[0], cy, anchor[2])
            elif axis == 'z':
                self.cursor_position = (anchor[0], anchor[1], cz)
            try:
                x, y, z = self.cursor_position
                self.voxelHovered.emit(x, y, z)
            except Exception:
                pass
            self.update()

        # Emit the set of active locks
        try:
            self.axisLockChanged.emit(tuple(sorted(self._axis_locks)))
        except Exception:
            pass

    def toggle_axis_lock(self, axis: str):
        """Toggle the axis lock for the given axis (supports multiple locks)."""
        if axis in self._axis_locks:
            self.clear_axis_lock(axis)
        else:
            self.set_axis_lock(axis)

    def clear_axis_lock(self, axis: str | None = None):
        """Clear axis lock(s). If axis provided, clear only that axis; otherwise clear all."""
        if axis is None:
            if self._axis_locks:
                print(f"Axis locks cleared: {','.join(sorted(a.upper() for a in self._axis_locks))}")
            self._axis_locks.clear()
            self._axis_lock_anchors.clear()
            self.update()
            try:
                self.axisLockChanged.emit(None)
            except Exception:
                pass
            return

        if axis in self._axis_locks:
            self._axis_locks.remove(axis)
            anchor = self._axis_lock_anchors.pop(axis, None)
            print(f"Axis lock cleared: {axis.upper()}")
            self.update()
            try:
                if self._axis_locks:
                    self.axisLockChanged.emit(tuple(sorted(self._axis_locks)))
                else:
                    self.axisLockChanged.emit(None)
            except Exception:
                pass

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press including axis locking and viewport toggles.

        Ignores autorepeat events and tracks pressed keys to avoid spurious
        set/clear calls when the OS or windowing system generates repeated
        key events.
        """
        try:
            print(f"Viewport keyPressEvent: key={event.key()}, autorepeat={event.isAutoRepeat()}")
        except Exception:
            print("Viewport keyPressEvent received")

        key = event.key()

        # Axis locking keys - toggle on non-autorepeat press (supports multiple locks)
        if key in (Qt.Key.Key_X, Qt.Key.Key_Y, Qt.Key.Key_Z):
            if event.isAutoRepeat():
                return
            axis = 'x' if key == Qt.Key.Key_X else ('y' if key == Qt.Key.Key_Y else 'z')
            self.toggle_axis_lock(axis)
            return

        # Other viewport toggles
        if key == Qt.Key.Key_Home:
            self.camera.reset()
        elif key == Qt.Key.Key_G:
            self.renderer.show_grid = not self.renderer.show_grid
        elif key == Qt.Key.Key_A:
            self.renderer.show_axes = not self.renderer.show_axes
        elif key == Qt.Key.Key_W:
            self.renderer.show_wireframe = not self.renderer.show_wireframe
        elif key == Qt.Key.Key_C:
            self.show_cursor = not self.show_cursor
            self.update()

        # Default behavior
        super().keyPressEvent(event)
        self.update()

    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key releases, clearing axis locks when appropriate."""
        try:
            print(f"Viewport keyReleaseEvent: key={event.key()}, autorepeat={event.isAutoRepeat()}")
        except Exception:
            print("Viewport keyReleaseEvent received")

        key = event.key()
        # Ignore autorepeat releases
        if event.isAutoRepeat():
            return

        # Release no longer clears axis locks (toggle mode); keep default behavior for other keys
        super().keyReleaseEvent(event)
    
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
        """Capture a screenshot of the viewport (accounts for HiDPI scaling)."""
        if not HAS_OPENGL:
            return None

        self.makeCurrent()
        width, height = self.width(), self.height()

        try:
            scale = float(self.devicePixelRatioF())
        except Exception:
            try:
                scale = float(self.devicePixelRatio())
            except Exception:
                scale = 1.0

        pix_w = max(1, int(round(width * scale)))
        pix_h = max(1, int(round(height * scale)))

        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, pix_w, pix_h, GL_RGBA, GL_UNSIGNED_BYTE)

        image = np.frombuffer(pixels, dtype=np.uint8)
        # Reshape into (height, width, 4) using pixel sizes (note the order: rows then cols)
        try:
            image = image.reshape(pix_h, pix_w, 4)
        except Exception:
            return None

        image = np.flipud(image)  # Flip vertically

        return image

    def _render_overlay(self):
        """Draw a small 2D overlay with axis-lock info."""
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Semi-transparent background
            rect_w, rect_h = 240, 56
            margin = 8
            painter.fillRect(margin, margin, rect_w, rect_h, QColor(0, 0, 0, 140))

            # Text
            painter.setPen(QColor(255, 255, 255))
            font = QFont('Sans', 10)
            painter.setFont(font)

            lock_text = 'Axis Lock: None'
            anchor_text = ''
            if self._axis_locks:
                lock_text = 'Axis Lock: ' + ','.join([a.upper() for a in sorted(self._axis_locks)])
                # Build per-axis anchor text
                anchors = []
                for a in sorted(self._axis_locks):
                    anch = self._axis_lock_anchors.get(a)
                    if anch is not None:
                        anchors.append(f"{a.upper()}: {anch[0]},{anch[1]},{anch[2]}")
                anchor_text = '  '.join(anchors)

            painter.drawText(margin + 8, margin + 18, lock_text)
            painter.drawText(margin + 8, margin + 36, anchor_text)

            painter.end()
        except Exception:
            pass
    
    def _render_cursor(self):
        """Render the 3D cursor at the current position."""
        if not HAS_OPENGL or self.cursor_position is None:
            return
        
        x, y, z = self.cursor_position
        
        # Map model cursor to world/render coordinates
        wxm, wym, wzm = self._model_to_world(x, y, z)

        if self.model is not None:
            rsx, rsy, rsz = self._get_render_size()
            offset_x = -rsx / 2
            offset_y = -rsy / 2
            offset_z = -rsz / 2
        else:
            offset_x = offset_y = offset_z = 0

        wx = wxm + offset_x
        wy = wym + offset_y
        wz = wzm + offset_z        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)  # Always show cursor on top
        
        # Draw wireframe cube with pulsing effect
        import time
        pulse = (math.sin(time.time() * 4) + 1) / 2  # 0 to 1
        brightness = 0.7 + pulse * 0.3  # 0.7 to 1.0
        
        glColor3f(brightness, brightness, 0.0)  # Yellow cursor with pulsing
        glLineWidth(4.0)
        
        # Make cursor slightly larger than a voxel
        expand = 0.05
        
        # Bottom face
        glBegin(GL_LINE_LOOP)
        glVertex3f(wx - expand, wy - expand, wz - expand)
        glVertex3f(wx + 1 + expand, wy - expand, wz - expand)
        glVertex3f(wx + 1 + expand, wy - expand, wz + 1 + expand)
        glVertex3f(wx - expand, wy - expand, wz + 1 + expand)
        glEnd()
        
        # Top face
        glBegin(GL_LINE_LOOP)
        glVertex3f(wx - expand, wy + 1 + expand, wz - expand)
        glVertex3f(wx + 1 + expand, wy + 1 + expand, wz - expand)
        glVertex3f(wx + 1 + expand, wy + 1 + expand, wz + 1 + expand)
        glVertex3f(wx - expand, wy + 1 + expand, wz + 1 + expand)
        glEnd()
        
        # Vertical edges
        glBegin(GL_LINES)
        glVertex3f(wx - expand, wy - expand, wz - expand)
        glVertex3f(wx - expand, wy + 1 + expand, wz - expand)
        glVertex3f(wx + 1 + expand, wy - expand, wz - expand)
        glVertex3f(wx + 1 + expand, wy + 1 + expand, wz - expand)
        glVertex3f(wx + 1 + expand, wy - expand, wz + 1 + expand)
        glVertex3f(wx + 1 + expand, wy + 1 + expand, wz + 1 + expand)
        glVertex3f(wx - expand, wy - expand, wz + 1 + expand)
        glVertex3f(wx - expand, wy + 1 + expand, wz + 1 + expand)
        glEnd()
        
        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def closeEvent(self, event):
        """Clean up on close."""
        self._update_timer.stop()
        self.renderer.cleanup()
        super().closeEvent(event)
