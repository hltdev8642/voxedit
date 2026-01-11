"""
MainWindow - Primary Application Window
=======================================

Main application window with menus, toolbars, dock widgets,
and central 3D viewport for the VoxEdit editor.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import math

from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QMenuBar, QMenu, QToolBar,
    QDockWidget, QFileDialog, QMessageBox, QStatusBar,
    QProgressDialog, QInputDialog, QWidget, QVBoxLayout,
    QLabel, QSplitter
)
from PyQt6.QtCore import Qt, QSettings, QTimer, pyqtSignal, QEvent
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QCloseEvent

from voxedit.core.voxel_model import VoxelModel
from voxedit.core.palette import VoxelPalette
from voxedit.core.operations import VoxelOperations
from voxedit.formats import FormatManager
from voxedit.gui.viewport import VoxelViewport
from voxedit.gui.tool_panel import ToolPanel, Tool
from voxedit.gui.palette_panel import PalettePanel


class MainWindow(QMainWindow):
    """
    Main application window for VoxEdit.
    
    Provides the primary user interface including:
    - Menu bar with file, edit, view, tools menus
    - Toolbars for quick access to common operations
    - Dock widgets for tools and palette
    - Central 3D viewport for voxel editing
    - Status bar for information display
    """
    
    modelChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("VoxEdit - Teardown Voxel Editor")
        self.setMinimumSize(1280, 720)
        
        # Initialize data
        self.model: Optional[VoxelModel] = None
        self.palette: Optional[VoxelPalette] = None
        self.operations: Optional[VoxelOperations] = None
        self.format_manager = FormatManager()
        
        self.current_file: Optional[str] = None
        self.modified = False
        
        # Undo/Redo stacks
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo = 50
        
        # Settings
        self.settings = QSettings("VoxEdit", "VoxEdit")
        
        # Create UI components
        self._create_viewport()
        self._create_panels()
        self._create_menus()
        self._create_toolbars()
        self._create_statusbar()
        self._create_axis_shortcuts()
        
        # Load settings
        self._load_settings()

        # Install an application-level event filter to catch raw key press/release events
        try:
            app = QApplication.instance()
            if app is not None:
                app.installEventFilter(self)
        except Exception:
            pass

        # Track pressed modifier/axis keys to ignore autorepeat noise
        self._pressed_keys = set()
        # Track which axis key presses we've already handled in the eventFilter to avoid duplicate toggles
        self._handled_axis_presses = set()
        
        # Create new empty model
        self.new_model()
    
    def _create_viewport(self):
        """Create the central 3D viewport."""
        self.viewport = VoxelViewport(self)
        self.viewport.voxelClicked.connect(self._on_voxel_clicked)
        self.viewport.voxelHovered.connect(self._on_voxel_hovered)
        self.viewport.toolDragStarted.connect(self._on_tool_drag_started)
        self.viewport.toolDragEnded.connect(self._on_tool_drag_ended)
        self.viewport.axisLockChanged.connect(self._on_axis_lock_changed)
        self._dragging = False
        self._tool_start = None
        self.setCentralWidget(self.viewport)

    def keyPressEvent(self, event):
        """Forward key press events to viewport for axis lock toggling (X/Y/Z)."""
        try:
            print(f"MainWindow keyPressEvent: key={event.key()}, autorepeat={event.isAutoRepeat()}, focus={self.focusWidget().__class__.__name__ if self.focusWidget() else None}")
        except Exception:
            print("MainWindow keyPressEvent")

        key = event.key()
        # Ignore autorepeat noise
        if event.isAutoRepeat():
            return

        if key == Qt.Key.Key_X:
            self.viewport.toggle_axis_lock('x')
            self.statusbar.showMessage(f"Toggled axis X: {'ON' if 'x' in self.viewport._axis_locks else 'OFF'}", 1500)
            return
        elif key == Qt.Key.Key_Y:
            self.viewport.toggle_axis_lock('y')
            self.statusbar.showMessage(f"Toggled axis Y: {'ON' if 'y' in self.viewport._axis_locks else 'OFF'}", 1500)
            return
        elif key == Qt.Key.Key_Z:
            self.viewport.toggle_axis_lock('z')
            self.statusbar.showMessage(f"Toggled axis Z: {'ON' if 'z' in self.viewport._axis_locks else 'OFF'}", 1500)
            return

        super().keyPressEvent(event)
    def keyReleaseEvent(self, event):
        """Forward key release events to viewport for axis lock clearing.

        Ignores autorepeat and only clears locks for keys we previously saw
        pressed.
        """
        try:
            print(f"MainWindow keyReleaseEvent: key={event.key()}, autorepeat={event.isAutoRepeat()}, focus={self.focusWidget().__class__.__name__ if self.focusWidget() else None}")
        except Exception:
            print("MainWindow keyReleaseEvent")

        # Ignore autorepeat noise
        if event.isAutoRepeat():
            return

        key = event.key()
        # In toggle mode, releasing X/Y/Z does not clear locks — keep default behavior
        super().keyReleaseEvent(event)
    
    def _create_panels(self):
        """Create dock widget panels."""
        # Tool Panel
        self.tool_dock = QDockWidget("Tools", self)
        self.tool_dock.setObjectName("ToolsDock")
        self.tool_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | 
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.tool_panel = ToolPanel(self)
        self.tool_dock.setWidget(self.tool_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.tool_dock)
        
        # Palette Panel
        self.palette_dock = QDockWidget("Palette", self)
        self.palette_dock.setObjectName("PaletteDock")
        self.palette_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | 
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.palette_panel = PalettePanel(self)
        self.palette_panel.colorSelected.connect(self._on_color_selected)
        self.palette_dock.setWidget(self.palette_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.palette_dock)
        
        # Model Info Panel
        self.info_dock = QDockWidget("Model Info", self)
        self.info_dock.setObjectName("InfoDock")
        self.info_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | 
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._create_info_panel()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.info_dock)
    
    def _create_info_panel(self):
        """Create the model information panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        
        self.info_size_label = QLabel("Size: -")
        self.info_voxels_label = QLabel("Voxels: -")
        self.info_file_label = QLabel("File: New")
        
        layout.addWidget(self.info_size_label)
        layout.addWidget(self.info_voxels_label)
        layout.addWidget(self.info_file_label)
        layout.addStretch()
        
        self.info_dock.setWidget(widget)

    def _create_axis_shortcuts(self):
        """Create fallback application-level shortcuts to toggle axis locks (Alt+X/Y/Z).
        Uses application-scoped QActions to ensure reliability across platforms."""
        act_x = QAction("Toggle Axis X", self)
        act_x.setShortcut(QKeySequence("Alt+X"))
        act_x.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        act_x.triggered.connect(lambda: self._toggle_axis_lock('x'))
        self.addAction(act_x)

        act_y = QAction("Toggle Axis Y", self)
        act_y.setShortcut(QKeySequence("Alt+Y"))
        act_y.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        act_y.triggered.connect(lambda: self._toggle_axis_lock('y'))
        self.addAction(act_y)

        act_z = QAction("Toggle Axis Z", self)
        act_z.setShortcut(QKeySequence("Alt+Z"))
        act_z.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        act_z.triggered.connect(lambda: self._toggle_axis_lock('z'))
        self.addAction(act_z)

    def _toggle_axis_lock(self, axis: str):
        """Toggle axis lock on the viewport as a fallback when hold-release events aren't reliable."""
        if not hasattr(self, 'viewport') or self.viewport is None:
            return
        self.viewport.toggle_axis_lock(axis)
        state = 'ON' if axis in self.viewport._axis_locks else 'OFF'
        self.statusbar.showMessage(f"Axis {axis.upper()} toggled: {state}", 1500)

    def eventFilter(self, obj, event):
        """Application-level event filter to capture key press/release (works even if focus is odd)."""
        if event.type() == QEvent.Type.KeyPress:
            try:
                print(f"EventFilter KeyPress: key={event.key()}, autorepeat={event.isAutoRepeat()}, obj={obj.__class__.__name__}")
            except Exception:
                print("EventFilter KeyPress")
            # Toggle on first non-autorepeat press, but avoid duplicate toggles for the same
            # physical press when the event is delivered to multiple Qt objects.
            if not event.isAutoRepeat():
                key = event.key()
                if key == Qt.Key.Key_X:
                    if 'x' not in self._handled_axis_presses:
                        self._handled_axis_presses.add('x')
                        self.viewport.toggle_axis_lock('x')
                    return True
                if key == Qt.Key.Key_Y:
                    if 'y' not in self._handled_axis_presses:
                        self._handled_axis_presses.add('y')
                        self.viewport.toggle_axis_lock('y')
                    return True
                if key == Qt.Key.Key_Z:
                    if 'z' not in self._handled_axis_presses:
                        self._handled_axis_presses.add('z')
                        self.viewport.toggle_axis_lock('z')
                    return True
        elif event.type() == QEvent.Type.KeyRelease:
            try:
                print(f"EventFilter KeyRelease: key={event.key()}, autorepeat={event.isAutoRepeat()}, obj={obj.__class__.__name__}")
            except Exception:
                print("EventFilter KeyRelease")
            # Clear our handled marker on final release so the next press can toggle again
            if not event.isAutoRepeat():
                key = event.key()
                if key == Qt.Key.Key_X and 'x' in self._handled_axis_presses:
                    self._handled_axis_presses.remove('x')
                if key == Qt.Key.Key_Y and 'y' in self._handled_axis_presses:
                    self._handled_axis_presses.remove('y')
                if key == Qt.Key.Key_Z and 'z' in self._handled_axis_presses:
                    self._handled_axis_presses.remove('z')

        return super().eventFilter(obj, event)        
        # Edit menu
    def _create_menus(self):
        """Create the application menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        self.action_new = QAction("&New", self)
        self.action_new.setShortcut(QKeySequence.StandardKey.New)
        self.action_new.triggered.connect(lambda: self.new_model())
        file_menu.addAction(self.action_new)
        
        self.action_open = QAction("&Open...", self)
        self.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open.triggered.connect(self.open_file)
        file_menu.addAction(self.action_open)
        
        # Recent files submenu
        self.recent_menu = file_menu.addMenu("Open &Recent")
        self._update_recent_files()
        
        file_menu.addSeparator()
        
        self.action_save = QAction("&Save", self)
        self.action_save.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save.triggered.connect(self.save_file)
        file_menu.addAction(self.action_save)
        
        self.action_save_as = QAction("Save &As...", self)
        self.action_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.action_save_as.triggered.connect(self.save_file_as)
        file_menu.addAction(self.action_save_as)
        
        file_menu.addSeparator()
        
        # Import submenu
        import_menu = file_menu.addMenu("&Import")
        
        self.action_import_vox = QAction("MagicaVoxel (.vox)...", self)
        self.action_import_vox.triggered.connect(lambda: self._import_format("vox"))
        import_menu.addAction(self.action_import_vox)
        
        self.action_import_schematic = QAction("Minecraft Schematic (.schematic)...", self)
        self.action_import_schematic.triggered.connect(lambda: self._import_format("schematic"))
        import_menu.addAction(self.action_import_schematic)
        
        self.action_import_schem = QAction("Minecraft Schem (.schem)...", self)
        self.action_import_schem.triggered.connect(lambda: self._import_format("schem"))
        import_menu.addAction(self.action_import_schem)
        
        self.action_import_binvox = QAction("Binvox (.binvox)...", self)
        self.action_import_binvox.triggered.connect(lambda: self._import_format("binvox"))
        import_menu.addAction(self.action_import_binvox)
        
        self.action_import_qb = QAction("Qubicle (.qb)...", self)
        self.action_import_qb.triggered.connect(lambda: self._import_format("qb"))
        import_menu.addAction(self.action_import_qb)
        
        # Export submenu
        export_menu = file_menu.addMenu("&Export")
        
        self.action_export_vox = QAction("MagicaVoxel (.vox)...", self)
        self.action_export_vox.triggered.connect(lambda: self._export_format("vox"))
        export_menu.addAction(self.action_export_vox)
        
        self.action_export_obj = QAction("Wavefront OBJ (.obj)...", self)
        self.action_export_obj.triggered.connect(lambda: self._export_format("obj"))
        export_menu.addAction(self.action_export_obj)
        
        self.action_export_stl = QAction("STL (.stl)...", self)
        self.action_export_stl.triggered.connect(lambda: self._export_format("stl"))
        export_menu.addAction(self.action_export_stl)
        
        self.action_export_ply = QAction("PLY (.ply)...", self)
        self.action_export_ply.triggered.connect(lambda: self._export_format("ply"))
        export_menu.addAction(self.action_export_ply)
        
        self.action_export_gltf = QAction("GLTF (.gltf)...", self)
        self.action_export_gltf.triggered.connect(lambda: self._export_format("gltf"))
        export_menu.addAction(self.action_export_gltf)
        
        self.action_export_schematic = QAction("Minecraft Schematic (.schematic)...", self)
        self.action_export_schematic.triggered.connect(lambda: self._export_format("schematic"))
        export_menu.addAction(self.action_export_schematic)
        
        file_menu.addSeparator()
        
        self.action_exit = QAction("E&xit", self)
        self.action_exit.setShortcut(QKeySequence.StandardKey.Quit)
        self.action_exit.triggered.connect(self.close)
        file_menu.addAction(self.action_exit)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        self.action_undo = QAction("&Undo", self)
        self.action_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.action_undo.triggered.connect(self.undo)
        edit_menu.addAction(self.action_undo)
        
        self.action_redo = QAction("&Redo", self)
        self.action_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.action_redo.triggered.connect(self.redo)
        edit_menu.addAction(self.action_redo)
        
        edit_menu.addSeparator()
        
        self.action_clear = QAction("&Clear All", self)
        self.action_clear.triggered.connect(self.clear_model)
        edit_menu.addAction(self.action_clear)
        
        edit_menu.addSeparator()
        
        self.action_resize = QAction("&Resize Model...", self)
        self.action_resize.triggered.connect(self.resize_model)
        edit_menu.addAction(self.action_resize)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        self.action_show_grid = QAction("Show &Grid", self)
        self.action_show_grid.setCheckable(True)
        self.action_show_grid.setChecked(True)
        self.action_show_grid.triggered.connect(self._toggle_grid)
        view_menu.addAction(self.action_show_grid)
        
        self.action_show_axes = QAction("Show &Axes", self)
        self.action_show_axes.setCheckable(True)
        self.action_show_axes.setChecked(True)
        self.action_show_axes.triggered.connect(self._toggle_axes)
        view_menu.addAction(self.action_show_axes)

        # Allow swapping Y and Z axes if a model is authored with different convention
        self.action_swap_yz = QAction("Swap Y/Z Axes", self)
        self.action_swap_yz.setCheckable(True)
        self.action_swap_yz.setChecked(False)
        self.action_swap_yz.triggered.connect(self._toggle_swap_yz)
        view_menu.addAction(self.action_swap_yz)
        
        self.action_wireframe = QAction("&Wireframe Mode", self)
        self.action_wireframe.setCheckable(True)
        self.action_wireframe.setShortcut("W")
        self.action_wireframe.triggered.connect(self._toggle_wireframe)
        view_menu.addAction(self.action_wireframe)
        
        view_menu.addSeparator()
        
        self.action_show_cursor = QAction("Show &Cursor", self)
        self.action_show_cursor.setCheckable(True)
        self.action_show_cursor.setChecked(True)
        self.action_show_cursor.setShortcut("C")
        self.action_show_cursor.triggered.connect(self._toggle_cursor)
        view_menu.addAction(self.action_show_cursor)
        
        view_menu.addSeparator()
        
        self.action_reset_view = QAction("&Reset View", self)
        self.action_reset_view.setShortcut("Home")
        self.action_reset_view.triggered.connect(self._reset_view)
        view_menu.addAction(self.action_reset_view)
        
        view_menu.addSeparator()
        
        # Swap Y/Z handler inserted
        def _toggle_swap_yz_local(checked: bool):
            if hasattr(self, 'viewport') and self.viewport is not None:
                self.viewport.swap_yz = checked
                self.viewport.rebuild_mesh()
                self.viewport.update()
                self.statusbar.showMessage(f"Swap Y/Z set to {checked}", 1500)
        self._toggle_swap_yz = _toggle_swap_yz_local

        # Dock visibility
        self.action_show_tools = self.tool_dock.toggleViewAction()
        self.action_show_tools.setText("&Tools Panel")
        view_menu.addAction(self.action_show_tools)
        
        self.action_show_palette = self.palette_dock.toggleViewAction()
        self.action_show_palette.setText("&Palette Panel")
        view_menu.addAction(self.action_show_palette)
        
        self.action_show_info = self.info_dock.toggleViewAction()
        self.action_show_info.setText("&Info Panel")
        view_menu.addAction(self.action_show_info)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        self.action_fill = QAction("&Fill Selection", self)
        self.action_fill.triggered.connect(self._tool_fill)
        tools_menu.addAction(self.action_fill)
        
        self.action_replace = QAction("&Replace Color...", self)
        self.action_replace.triggered.connect(self._tool_replace)
        tools_menu.addAction(self.action_replace)
        
        tools_menu.addSeparator()
        
        self.action_rotate_x = QAction("Rotate X 90°", self)
        self.action_rotate_x.triggered.connect(lambda: self._rotate_model('x', 90))
        tools_menu.addAction(self.action_rotate_x)
        
        self.action_rotate_y = QAction("Rotate Y 90°", self)
        self.action_rotate_y.triggered.connect(lambda: self._rotate_model('y', 90))
        tools_menu.addAction(self.action_rotate_y)
        
        self.action_rotate_z = QAction("Rotate Z 90°", self)
        self.action_rotate_z.triggered.connect(lambda: self._rotate_model('z', 90))
        tools_menu.addAction(self.action_rotate_z)
        
        tools_menu.addSeparator()
        
        self.action_mirror_x = QAction("Mirror X", self)
        self.action_mirror_x.triggered.connect(lambda: self._mirror_model('x'))
        tools_menu.addAction(self.action_mirror_x)
        
        self.action_mirror_y = QAction("Mirror Y", self)
        self.action_mirror_y.triggered.connect(lambda: self._mirror_model('y'))
        tools_menu.addAction(self.action_mirror_y)
        
        self.action_mirror_z = QAction("Mirror Z", self)
        self.action_mirror_z.triggered.connect(lambda: self._mirror_model('z'))
        tools_menu.addAction(self.action_mirror_z)
        
        tools_menu.addSeparator()
        
        self.action_hollow = QAction("&Hollow", self)
        self.action_hollow.triggered.connect(self._tool_hollow)
        tools_menu.addAction(self.action_hollow)
        
        self.action_smooth = QAction("&Smooth", self)
        self.action_smooth.triggered.connect(self._tool_smooth)
        tools_menu.addAction(self.action_smooth)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        self.action_about = QAction("&About VoxEdit", self)
        self.action_about.triggered.connect(self._show_about)
        help_menu.addAction(self.action_about)
    
    def _create_toolbars(self):
        """Create the application toolbars."""
        # File toolbar
        file_toolbar = QToolBar("File", self)
        file_toolbar.setObjectName("FileToolbar")
        self.addToolBar(file_toolbar)
        
        file_toolbar.addAction(self.action_new)
        file_toolbar.addAction(self.action_open)
        file_toolbar.addAction(self.action_save)
        
        # Edit toolbar
        edit_toolbar = QToolBar("Edit", self)
        edit_toolbar.setObjectName("EditToolbar")
        self.addToolBar(edit_toolbar)
        
        edit_toolbar.addAction(self.action_undo)
        edit_toolbar.addAction(self.action_redo)
        
        # View toolbar
        view_toolbar = QToolBar("View", self)
        view_toolbar.setObjectName("ViewToolbar")
        self.addToolBar(view_toolbar)
        
        view_toolbar.addAction(self.action_show_grid)
        view_toolbar.addAction(self.action_show_axes)
        view_toolbar.addAction(self.action_wireframe)
        view_toolbar.addAction(self.action_reset_view)
    
    def _create_statusbar(self):
        """Create the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Permanent widgets
        self.status_voxels = QLabel("Voxels: 0")
        self.statusbar.addPermanentWidget(self.status_voxels)
        
        self.status_size = QLabel("Size: 0×0×0")
        self.statusbar.addPermanentWidget(self.status_size)
        
        self.status_cursor = QLabel("Cursor: (0,0,0)")
        self.statusbar.addPermanentWidget(self.status_cursor)

        # Axis lock indicator
        self.status_axis = QLabel("")
        self.statusbar.addPermanentWidget(self.status_axis)
    
    def _update_window_title(self):
        """Update the window title based on current state."""
        title = "VoxEdit"
        
        if self.current_file:
            title = f"{Path(self.current_file).name} - VoxEdit"
        else:
            title = "Untitled - VoxEdit"
        
        if self.modified:
            title = f"*{title}"
        
        self.setWindowTitle(title)
    
    def _update_info(self):
        """Update the model info panel."""
        if self.model:
            size = self.model.size
            self.info_size_label.setText(f"Size: {size[0]}×{size[1]}×{size[2]}")
            
            voxel_count = self.model.get_voxel_count()
            self.info_voxels_label.setText(f"Voxels: {voxel_count:,}")
            
            self.status_size.setText(f"Size: {size[0]}×{size[1]}×{size[2]}")
            self.status_voxels.setText(f"Voxels: {voxel_count:,}")
        
        if self.current_file:
            self.info_file_label.setText(f"File: {Path(self.current_file).name}")
        else:
            self.info_file_label.setText("File: New")

    def _on_axis_lock_changed(self, axis):
        """Update status bar axis lock indicator when viewport changes lock state.

        `axis` may be None or an iterable of axis chars (e.g., ('x','y')) when
        multiple locks are active.
        """
        if not axis:
            self.status_axis.setText("")
            return
        # Accept tuple/list/set or single value
        if isinstance(axis, (list, tuple, set)):
            parts = [f"[{a.upper()}]" for a in axis]
            self.status_axis.setText(''.join(parts))
        else:
            try:
                self.status_axis.setText(f"[{axis.upper()}]")
            except Exception:
                self.status_axis.setText("")
    
    def _update_recent_files(self):
        """Update the recent files menu."""
        self.recent_menu.clear()
        
        recent = self.settings.value("recent_files", [])
        if not recent:
            action = self.recent_menu.addAction("(No recent files)")
            action.setEnabled(False)
            return
        
        for filepath in recent[:10]:
            if os.path.exists(filepath):
                action = self.recent_menu.addAction(Path(filepath).name)
                action.setData(filepath)
                action.triggered.connect(lambda checked, f=filepath: self.open_file(f))
    
    def _add_recent_file(self, filepath: str):
        """Add a file to the recent files list."""
        recent = self.settings.value("recent_files", [])
        
        if filepath in recent:
            recent.remove(filepath)
        
        recent.insert(0, filepath)
        recent = recent[:10]
        
        self.settings.setValue("recent_files", recent)
        self._update_recent_files()
    
    def _set_modified(self, modified: bool = True):
        """Set the modified flag."""
        self.modified = modified
        self._update_window_title()
    
    def _push_undo(self):
        """Save current state to undo stack."""
        if self.model is None:
            return
        
        state = {
            'voxels': self.model.voxels.copy(),
            'size': self.model.size
        }
        
        self.undo_stack.append(state)
        
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)
        
        self.redo_stack.clear()
        self._set_modified()
    
    # ==================== File Operations ====================
    
    def new_model(self, size: tuple = (32, 32, 32)):
        """Create a new empty model."""
        if self.modified:
            if not self._confirm_discard():
                return
        
        self.model = VoxelModel(size=size)
        self.palette = VoxelPalette()
        self.operations = VoxelOperations(self.model)
        
        # Create a test model with some voxels for demonstration
        self._create_demo_model()
        
        self.current_file = None
        self.modified = False
        
        self.undo_stack.clear()
        self.redo_stack.clear()
        
        self.viewport.set_model(self.model, self.palette)
        self.palette_panel.set_palette(self.palette)
        
        self._update_window_title()
        self._update_info()
        self.statusbar.showMessage("Created new model", 3000)
    
    def _create_demo_model(self):
        """Create a demonstration model with various shapes."""
        if self.model is None:
            return
        
        # Clear the model first
        self.model.clear()
        
        # Create a colorful cube in the center
        center_x, center_y, center_z = 16, 16, 16
        
        # Main cube (4x4x4)
        for x in range(center_x - 2, center_x + 2):
            for y in range(center_y - 2, center_y + 2):
                for z in range(center_z - 2, center_z + 2):
                    if self.model.is_valid_position(x, y, z):
                        self.model.set_voxel(x, y, z, 1)  # Red
        
        # Add some details - smaller cubes on corners
        corners = [
            (center_x - 3, center_y - 3, center_z - 3),
            (center_x + 3, center_y - 3, center_z - 3),
            (center_x - 3, center_y + 3, center_z - 3),
            (center_x + 3, center_y + 3, center_z - 3),
            (center_x - 3, center_y - 3, center_z + 3),
            (center_x + 3, center_y - 3, center_z + 3),
            (center_x - 3, center_y + 3, center_z + 3),
            (center_x + 3, center_y + 3, center_z + 3),
        ]
        
        for corner in corners:
            x, y, z = corner
            if self.model.is_valid_position(x, y, z):
                self.model.set_voxel(x, y, z, 2)  # Blue
        
        # Add a pillar
        for y in range(center_y - 4, center_y + 4):
            if self.model.is_valid_position(center_x, y, center_z):
                self.model.set_voxel(center_x, y, center_z, 3)  # Green
        
        # Set up some colors in the palette
        self.palette.set_color(1, 255, 100, 100)  # Red
        self.palette.set_color(2, 100, 100, 255)  # Blue
        self.palette.set_color(3, 100, 255, 100)  # Green
        self.palette.set_color(4, 255, 255, 100)  # Yellow
    
    def open_file(self, filepath: str = None):
        """Open a voxel file."""
        if self.modified:
            if not self._confirm_discard():
                return
        
        if not filepath:
            filters = self.format_manager.get_import_filter_string()
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Open Voxel File", "", filters
            )
        
        if not filepath:
            return
        
        try:
            model, palette = self.format_manager.import_file(filepath)
            
            self.model = model
            self.palette = palette if palette else VoxelPalette()
            self.operations = VoxelOperations(self.model)
            
            self.current_file = filepath
            self.modified = False
            
            self.undo_stack.clear()
            self.redo_stack.clear()
            
            self.viewport.set_model(self.model, self.palette)
            self.palette_panel.set_palette(self.palette)
            
            self._add_recent_file(filepath)
            self._update_window_title()
            self._update_info()
            
            self.statusbar.showMessage(f"Opened: {filepath}", 3000)
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to open file:\n{str(e)}"
            )
    
    def save_file(self):
        """Save the current file."""
        if not self.current_file:
            return self.save_file_as()
        
        return self._save_to_file(self.current_file)
    
    def save_file_as(self):
        """Save the file with a new name."""
        filters = "MagicaVoxel (*.vox);;All Files (*)"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Voxel File", "", filters
        )
        
        if not filepath:
            return False
        
        return self._save_to_file(filepath)
    
    def _save_to_file(self, filepath: str) -> bool:
        """Save the model to a file."""
        try:
            self.format_manager.export_file(
                filepath, self.model, self.palette
            )
            
            self.current_file = filepath
            self.modified = False
            
            self._add_recent_file(filepath)
            self._update_window_title()
            
            self.statusbar.showMessage(f"Saved: {filepath}", 3000)
            return True
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to save file:\n{str(e)}"
            )
            return False
    
    def _import_format(self, fmt: str):
        """Import from a specific format."""
        filters = {
            'vox': "MagicaVoxel (*.vox)",
            'schematic': "Minecraft Schematic (*.schematic)",
            'schem': "Minecraft Schem (*.schem)",
            'binvox': "Binvox (*.binvox)",
            'qb': "Qubicle (*.qb)"
        }
        
        filepath, _ = QFileDialog.getOpenFileName(
            self, f"Import {fmt.upper()}", "",
            filters.get(fmt, "All Files (*)")
        )
        
        if filepath:
            self.open_file(filepath)
    
    def _export_format(self, fmt: str):
        """Export to a specific format."""
        filters = {
            'vox': "MagicaVoxel (*.vox)",
            'obj': "Wavefront OBJ (*.obj)",
            'stl': "STL (*.stl)",
            'ply': "PLY (*.ply)",
            'gltf': "GLTF (*.gltf)",
            'schematic': "Minecraft Schematic (*.schematic)"
        }
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, f"Export {fmt.upper()}", "",
            filters.get(fmt, "All Files (*)")
        )
        
        if filepath:
            try:
                self.format_manager.export_file(
                    filepath, self.model, self.palette
                )
                self.statusbar.showMessage(f"Exported: {filepath}", 3000)
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to export:\n{str(e)}"
                )
    
    def _confirm_discard(self) -> bool:
        """Ask user to confirm discarding changes."""
        result = QMessageBox.question(
            self, "Unsaved Changes",
            "You have unsaved changes. Do you want to discard them?",
            QMessageBox.StandardButton.Yes | 
            QMessageBox.StandardButton.No
        )
        return result == QMessageBox.StandardButton.Yes
    
    # ==================== Edit Operations ====================
    
    def undo(self):
        """Undo the last operation."""
        if not self.undo_stack:
            return
        
        # Save current state to redo stack
        current_state = {
            'voxels': self.model.voxels.copy(),
            'size': self.model.size
        }
        self.redo_stack.append(current_state)
        
        # Restore previous state
        state = self.undo_stack.pop()
        self.model.voxels = state['voxels']
        
        self.viewport.rebuild_mesh()
        self._update_info()
        self.statusbar.showMessage("Undo", 2000)
    
    def redo(self):
        """Redo the last undone operation."""
        if not self.redo_stack:
            return
        
        # Save current state to undo stack
        current_state = {
            'voxels': self.model.voxels.copy(),
            'size': self.model.size
        }
        self.undo_stack.append(current_state)
        
        # Restore state
        state = self.redo_stack.pop()
        self.model.voxels = state['voxels']
        
        self.viewport.rebuild_mesh()
        self._update_info()
        self.statusbar.showMessage("Redo", 2000)
    
    def clear_model(self):
        """Clear all voxels from the model."""
        result = QMessageBox.question(
            self, "Clear Model",
            "Are you sure you want to clear all voxels?",
            QMessageBox.StandardButton.Yes | 
            QMessageBox.StandardButton.No
        )
        
        if result == QMessageBox.StandardButton.Yes:
            self._push_undo()
            self.model.clear()
            self.viewport.rebuild_mesh()
            self._update_info()
    
    def resize_model(self):
        """Resize the model dimensions."""
        size, ok = QInputDialog.getText(
            self, "Resize Model",
            "Enter new size (width height depth):",
            text=f"{self.model.size[0]} {self.model.size[1]} {self.model.size[2]}"
        )
        
        if ok and size:
            try:
                parts = size.split()
                new_size = (int(parts[0]), int(parts[1]), int(parts[2]))
                
                if all(1 <= s <= 256 for s in new_size):
                    self._push_undo()
                    self.model.resize(new_size)
                    self.viewport.rebuild_mesh()
                    self._update_info()
                else:
                    QMessageBox.warning(
                        self, "Invalid Size",
                        "Size must be between 1 and 256"
                    )
            except (ValueError, IndexError):
                QMessageBox.warning(
                    self, "Invalid Input",
                    "Please enter three numbers separated by spaces"
                )
    
    # ==================== View Operations ====================
    
    def _toggle_grid(self, checked: bool):
        """Toggle grid visibility."""
        self.viewport.renderer.show_grid = checked
        self.viewport.update()
    
    def _toggle_axes(self, checked: bool):
        """Toggle axes visibility."""
        self.viewport.renderer.show_axes = checked
        self.viewport.update()

    def _toggle_swap_yz(self, checked: bool):
        """Toggle swapping Y and Z axes in the viewport (useful for differing model conventions)."""
        if hasattr(self, 'viewport') and self.viewport is not None:
            self.viewport.swap_yz = checked
            self.viewport.rebuild_mesh()
            self.viewport.update()
            self.statusbar.showMessage(f"Swap Y/Z set to {checked}", 1500)
    
    def _toggle_wireframe(self, checked: bool):
        """Toggle wireframe mode."""
        self.viewport.renderer.show_wireframe = checked
        self.viewport.update()
    
    def _toggle_cursor(self, checked: bool):
        """Toggle 3D cursor visibility."""
        self.viewport.show_cursor = checked
        self.viewport.update()
    
    def _reset_view(self):
        """Reset the camera view."""
        self.viewport.camera.reset()
        self.viewport.update()
    
    # ==================== Tool Operations ====================
    
    def _on_color_selected(self, color_index: int):
        """Handle color selection from palette."""
        self.tool_panel.set_current_color(color_index)
    
    def _on_voxel_hovered(self, x: int, y: int, z: int):
        """Handle voxel hover to update cursor position in status bar."""
        lock_suffix = ''
        try:
            locks = getattr(self.viewport, '_axis_locks', None)
            if locks:
                lock_suffix = ' ' + ''.join([f'[{a.upper()}]' for a in sorted(locks)])
        except Exception:
            lock_suffix = ''

        if x < 0:
            self.status_cursor.setText(f"Cursor: (-,-,-){lock_suffix}")
        else:
            self.status_cursor.setText(f"Cursor: ({x},{y},{z}){lock_suffix}")
    
    def _on_voxel_clicked(self, x: int, y: int, z: int, button: int):
        """Handle voxel clicking with current tool."""
        print(f"TOOL: _on_voxel_clicked called with ({x}, {y}, {z}), button={button}")
        
        if self.model is None or self.operations is None:
            print("TOOL: Model or operations is None")
            return
        
        # Get current tool and color
        tool = self.tool_panel.get_current_tool()
        color = self.tool_panel.get_current_color()
        
        print(f"TOOL: Using tool {tool}, color {color}")
        
        # If a drag was started, undo was already pushed once; otherwise push undo for this single action
        if not getattr(self, '_dragging', False):
            self._push_undo()

        # Brush/shape parameters
        brush_size = self.tool_panel.get_brush_size()
        brush_shape = self.tool_panel.get_brush_shape()
        mirror_axes = self.tool_panel.get_mirror_axes()

        if tool == Tool.PENCIL:
            print(f"TOOL: Applying pencil brush at ({x}, {y}, {z}) with color {color}, size {brush_size}")
            self._apply_brush((x, y, z), brush_size, brush_shape, color, mirror_axes, mode='set')
        elif tool == Tool.ERASER:
            print(f"TOOL: Applying eraser brush at ({x}, {y}, {z}), size {brush_size}")
            self._apply_brush((x, y, z), brush_size, brush_shape, 0, mirror_axes, mode='set')
        elif tool == Tool.PAINT:
            print(f"TOOL: Applying paint brush at ({x}, {y}, {z}) with color {color}, size {brush_size}")
            self._apply_brush((x, y, z), brush_size, brush_shape, color, mirror_axes, mode='paint')
        elif tool == Tool.EYEDROPPER:
            # Pick color from voxel
            picked_color = self.model.get_voxel(x, y, z)
            print(f"TOOL: Eyedropper picked color {picked_color} from ({x}, {y}, {z})")
            if picked_color != 0:
                self.tool_panel.set_current_color(picked_color)
                self.palette_panel.select_color(picked_color)
        elif tool == Tool.FILL:
            # Flood fill from clicked voxel
            print(f"TOOL: Applying fill tool at ({x}, {y}, {z}) with color {color}")
            self.model.flood_fill((x, y, z), color)
        elif tool in (Tool.LINE, Tool.BOX, Tool.SPHERE):
            # Shape tools: start/finish behavior
            if getattr(self, '_tool_start', None) is None:
                self._tool_start = (x, y, z)
                self.statusbar.showMessage(f"{tool.name} start set at {self._tool_start}", 2000)
            else:
                start = self._tool_start
                end = (x, y, z)
                print(f"TOOL: {tool.name} from {start} to {end} with color {color}")
                if tool == Tool.LINE:
                    self.operations.draw_line(start, end, value=color)
                elif tool == Tool.BOX:
                    self.operations.draw_box(start, end, value=color, filled=True)
                elif tool == Tool.SPHERE:
                    # Interpret distance as radius
                    radius = max(1, int(round(math.dist(start, end))))
                    self.operations.draw_sphere(end, radius, value=color)
                self._tool_start = None
        
        # If not dragging, commit/finish immediately
        if not getattr(self, '_dragging', False):
            self.model.commit()
            self.viewport.rebuild_mesh()
            self._update_info()
            self.modified = True
        else:
            # While dragging, just refresh visuals; final commit will happen on drag end
            self.viewport.rebuild_mesh()
            self._update_info()
            self.modified = True
    
    def _tool_fill(self):
        """Fill operation - placeholder."""
        self.statusbar.showMessage("Fill tool - select area to fill", 3000)

    def _apply_brush(self, center: Tuple[int, int, int], size: int, shape: str, value: int, mirror_axes: tuple, mode: str = 'set'):
        """Apply a brush centered at `center`.

        mode: 'set' = set to value, 'paint' = set to value only on existing voxels
        """
        if self.model is None:
            return

        cx, cy, cz = center
        radius = max(0, (size - 1) // 2)

        coords = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if shape == 'square':
                        ok = True
                    elif shape == 'circle':
                        ok = (dx*dx + dy*dy + dz*dz) <= (radius * radius)
                    elif shape == 'diamond':
                        ok = (abs(dx) + abs(dy) + abs(dz)) <= radius
                    else:
                        ok = True

                    if not ok:
                        continue

                    x = cx + dx
                    y = cy + dy
                    z = cz + dz

                    if not self.model.is_valid_position(x, y, z):
                        continue

                    coords.append((x, y, z))

                    # mirror axes
                    mx, my, mz = mirror_axes
                    if mx:
                        mxp = (cx - dx, y, z)
                        if self.model.is_valid_position(*mxp):
                            coords.append(mxp)
                    if my:
                        myp = (x, cy - dy, z)
                        if self.model.is_valid_position(*myp):
                            coords.append(myp)
                    if mz:
                        mzp = (x, y, cz - dz)
                        if self.model.is_valid_position(*mzp):
                            coords.append(mzp)

        # Deduplicate
        coords = list({(a,b,c) for a,b,c in coords})

        # Apply changes
        for x, y, z in coords:
            if mode == 'set':
                self.model.set_voxel(x, y, z, value)
            elif mode == 'paint':
                if self.model.get_voxel(x, y, z) != 0:
                    self.model.set_voxel(x, y, z, value)

        # If not dragging, commit immediately; otherwise visual update handled externally
        if not getattr(self, '_dragging', False):
            self.model.commit()
        self.viewport.rebuild_mesh()
        self._update_info()
    
    def _tool_replace(self):
        """Replace color operation."""
        if self.model is None:
            return
        
        from_color, ok = QInputDialog.getInt(
            self, "Replace Color",
            "Enter color index to replace (1-255):",
            value=1, min=1, max=255
        )
        
        if not ok:
            return
        
        to_color, ok = QInputDialog.getInt(
            self, "Replace Color",
            "Enter replacement color index (1-255):",
            value=1, min=1, max=255
        )
        
        if ok:
            self._push_undo()
            
            # Find and replace all voxels with from_color
            mask = self.model.voxels == from_color
            self.model.voxels[mask] = to_color
            
            self.viewport.rebuild_mesh()
            self._update_info()
            self.statusbar.showMessage(f"Replaced color {from_color} with {to_color}", 3000)
    
    def _rotate_model(self, axis: str, degrees: int):
        """Rotate the model around an axis."""
        if self.operations is None:
            return
        
        self._push_undo()
        self.operations.rotate(axis, degrees)
        self.viewport.rebuild_mesh()
        self._update_info()
        self.statusbar.showMessage(f"Rotated {degrees}° around {axis.upper()} axis", 3000)
    
    def _mirror_model(self, axis: str):
        """Mirror the model along an axis."""
        if self.operations is None:
            return
        
        self._push_undo()
        self.operations.mirror(axis)
        self.viewport.rebuild_mesh()
        self._update_info()
        self.statusbar.showMessage(f"Mirrored along {axis.upper()} axis", 3000)
    
    def _tool_hollow(self):
        """Hollow out the model."""
        if self.operations is None:
            return
        
        thickness, ok = QInputDialog.getInt(
            self, "Hollow",
            "Enter shell thickness:",
            value=1, min=1, max=10
        )
        
        if ok:
            self._push_undo()
            self.operations.shell(thickness)
            self.viewport.rebuild_mesh()
            self._update_info()
            self.statusbar.showMessage(f"Hollowed with thickness {thickness}", 3000)
    
    def _tool_smooth(self):
        """Smooth the model surface."""
        if self.operations is None:
            return
        
        self._push_undo()
        self.operations.smooth()
        self.viewport.rebuild_mesh()
        self._update_info()
        self.statusbar.showMessage("Applied smoothing", 3000)

    def _on_tool_drag_started(self):
        """Called when a drag operation starts in the viewport (begin grouped undo)."""
        # Push a single undo entry per drag
        if not getattr(self, '_dragging', False):
            self._push_undo()
            self._dragging = True
            self._drag_was_active = True

    def _on_tool_drag_ended(self):
        """Called when a drag operation ends in the viewport (finalize commit)."""
        if getattr(self, '_dragging', False):
            # Finalize changes made during drag
            if self.model is not None:
                self.model.commit()
            self.viewport.rebuild_mesh()
            self._update_info()
            self._dragging = False
            self._drag_was_active = False
    
    # ==================== Help ====================
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About VoxEdit",
            """<h2>VoxEdit</h2>
            <p>Teardown Voxel Editor</p>
            <p>Version 1.0.0</p>
            <p>A powerful voxel editor with support for multiple formats including:</p>
            <ul>
            <li>MagicaVoxel (.vox)</li>
            <li>Minecraft Schematics (.schematic, .schem)</li>
            <li>Mesh exports (OBJ, STL, PLY, GLTF)</li>
            <li>Binvox, Qubicle, and more</li>
            </ul>
            """
        )
    
    # ==================== Settings ====================
    
    def _load_settings(self):
        """Load application settings."""
        # Window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
    
    def _save_settings(self):
        """Save application settings."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
    
    def closeEvent(self, event: QCloseEvent):
        """Handle window close."""
        if self.modified:
            result = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save before exiting?",
                QMessageBox.StandardButton.Save | 
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )
            
            if result == QMessageBox.StandardButton.Save:
                if not self.save_file():
                    event.ignore()
                    return
            elif result == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
        
        self._save_settings()
        event.accept()
