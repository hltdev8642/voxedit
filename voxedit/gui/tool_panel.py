"""
ToolPanel - Editing Tools Panel
===============================

Dock widget panel containing editing tools for voxel manipulation.
"""

from typing import Optional, Callable
from enum import Enum, auto

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QToolButton, QButtonGroup, QLabel, QSlider,
    QSpinBox, QGroupBox, QFrame, QComboBox,
    QCheckBox, QPushButton, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QPointF
from PyQt6.QtGui import QIcon, QColor, QPainter, QPixmap, QPolygonF


class Tool(Enum):
    """Available editing tools."""
    SELECT = auto()
    PENCIL = auto()
    LINE = auto()
    BOX = auto()
    SPHERE = auto()
    FILL = auto()
    ERASER = auto()
    EYEDROPPER = auto()
    PAINT = auto()


class ToolButton(QToolButton):
    """Custom tool button with icon and tooltip."""
    
    def __init__(self, tool: Tool, tooltip: str, parent=None):
        super().__init__(parent)
        
        self.tool = tool
        self.setToolTip(tooltip)
        self.setCheckable(True)
        self.setFixedSize(48, 48)
        self.setIconSize(QSize(32, 32))
        
        # Create simple icon
        self._create_icon()
    
    def _create_icon(self):
        """Create a simple icon for the tool."""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw based on tool type
        color = QColor(100, 100, 100)
        painter.setPen(color)
        painter.setBrush(color)
        
        if self.tool == Tool.SELECT:
            # Selection rectangle
            painter.drawRect(4, 4, 24, 24)
        elif self.tool == Tool.PENCIL:
            # Pencil shape
            polygon = QPolygonF([
                QPointF(8, 24), QPointF(24, 8), QPointF(28, 12), QPointF(12, 28)
            ])
            painter.drawPolygon(polygon)
        elif self.tool == Tool.LINE:
            # Line
            painter.drawLine(4, 28, 28, 4)
        elif self.tool == Tool.BOX:
            # Filled box
            painter.fillRect(6, 6, 20, 20, color)
        elif self.tool == Tool.SPHERE:
            # Circle/sphere
            painter.drawEllipse(4, 4, 24, 24)
        elif self.tool == Tool.FILL:
            # Paint bucket
            painter.drawRect(8, 8, 16, 16)
            painter.drawLine(16, 8, 16, 24)
        elif self.tool == Tool.ERASER:
            # Eraser shape
            painter.drawRect(6, 10, 20, 12)
        elif self.tool == Tool.EYEDROPPER:
            # Dropper shape
            painter.drawEllipse(4, 4, 12, 12)
            painter.drawLine(12, 12, 24, 24)
        elif self.tool == Tool.PAINT:
            # Paint brush
            painter.drawRect(4, 16, 8, 12)
            painter.drawLine(8, 16, 24, 4)
        
        painter.end()
        
        self.setIcon(QIcon(pixmap))


class BrushSizeSlider(QWidget):
    """Slider widget for brush size control."""
    
    valueChanged = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        self.label = QLabel("Size:")
        layout.addWidget(self.label)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(20)
        self.slider.setValue(1)
        self.slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.slider)
        
        self.spinbox = QSpinBox()
        self.spinbox.setMinimum(1)
        self.spinbox.setMaximum(20)
        self.spinbox.setValue(1)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)
        self.spinbox.setFixedWidth(60)
        layout.addWidget(self.spinbox)
    
    def _on_value_changed(self, value: int):
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(value)
        self.spinbox.blockSignals(False)
        self.valueChanged.emit(value)
    
    def _on_spinbox_changed(self, value: int):
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)
    
    def value(self) -> int:
        return self.slider.value()
    
    def setValue(self, value: int):
        self.slider.setValue(value)


class ColorPreview(QWidget):
    """Widget showing current selected color."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.color = QColor(255, 255, 255)
        self.color_index = 1
        
        self.setMinimumSize(48, 48)
        self.setMaximumSize(64, 64)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    
    def set_color(self, color: QColor, index: int):
        self.color = color
        self.color_index = index
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw checkerboard for transparency
        checker_size = 8
        for i in range(0, self.width(), checker_size):
            for j in range(0, self.height(), checker_size):
                if (i // checker_size + j // checker_size) % 2 == 0:
                    painter.fillRect(i, j, checker_size, checker_size, 
                                    QColor(200, 200, 200))
                else:
                    painter.fillRect(i, j, checker_size, checker_size,
                                    QColor(255, 255, 255))
        
        # Draw color
        painter.fillRect(0, 0, self.width(), self.height(), self.color)
        
        # Draw border
        painter.setPen(QColor(100, 100, 100))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
        
        # Draw index text
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(4, 14, f"#{self.color_index}")


class ToolPanel(QWidget):
    """
    Panel containing editing tools and brush settings.
    
    Signals:
        toolChanged: Emitted when the selected tool changes
        brushSizeChanged: Emitted when brush size changes
    """
    
    toolChanged = pyqtSignal(Tool)
    brushSizeChanged = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_tool = Tool.PENCIL
        self.current_color = 1
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        
        # Tools group
        tools_group = QGroupBox("Tools")
        tools_layout = QGridLayout(tools_group)
        tools_layout.setSpacing(4)
        
        self.tool_buttons = QButtonGroup(self)
        self.tool_buttons.setExclusive(True)
        
        tools = [
            (Tool.SELECT, "Select (V)", 0, 0),
            (Tool.PENCIL, "Pencil (P)", 0, 1),
            (Tool.LINE, "Line (L)", 0, 2),
            (Tool.BOX, "Box (B)", 1, 0),
            (Tool.SPHERE, "Sphere (S)", 1, 1),
            (Tool.FILL, "Fill (F)", 1, 2),
            (Tool.ERASER, "Eraser (E)", 2, 0),
            (Tool.EYEDROPPER, "Eyedropper (I)", 2, 1),
            (Tool.PAINT, "Paint (G)", 2, 2),
        ]
        
        for tool, tooltip, row, col in tools:
            btn = ToolButton(tool, tooltip)
            btn.clicked.connect(lambda checked, t=tool: self._on_tool_clicked(t))
            self.tool_buttons.addButton(btn)
            tools_layout.addWidget(btn, row, col)
            
            if tool == Tool.PENCIL:
                btn.setChecked(True)
        
        layout.addWidget(tools_group)
        
        # Color preview group
        color_group = QGroupBox("Current Color")
        color_layout = QVBoxLayout(color_group)
        
        preview_layout = QHBoxLayout()
        self.color_preview = ColorPreview()
        preview_layout.addWidget(self.color_preview)
        preview_layout.addStretch()
        
        color_layout.addLayout(preview_layout)
        
        # Color index spinner
        index_layout = QHBoxLayout()
        index_layout.addWidget(QLabel("Index:"))
        self.color_spinbox = QSpinBox()
        self.color_spinbox.setMinimum(1)
        self.color_spinbox.setMaximum(255)
        self.color_spinbox.setValue(1)
        self.color_spinbox.valueChanged.connect(self._on_color_index_changed)
        index_layout.addWidget(self.color_spinbox)
        color_layout.addLayout(index_layout)
        
        layout.addWidget(color_group)
        
        # Brush settings group
        brush_group = QGroupBox("Brush Settings")
        brush_layout = QVBoxLayout(brush_group)
        
        # Brush size
        self.brush_size = BrushSizeSlider()
        self.brush_size.valueChanged.connect(self.brushSizeChanged.emit)
        brush_layout.addWidget(self.brush_size)
        
        # Brush shape
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(QLabel("Shape:"))
        self.brush_shape = QComboBox()
        self.brush_shape.addItems(["Square", "Circle", "Diamond"])
        shape_layout.addWidget(self.brush_shape)
        brush_layout.addLayout(shape_layout)
        
        layout.addWidget(brush_group)
        
        # Options group
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        
        self.mirror_x = QCheckBox("Mirror X")
        options_layout.addWidget(self.mirror_x)
        
        self.mirror_y = QCheckBox("Mirror Y")
        options_layout.addWidget(self.mirror_y)
        
        self.mirror_z = QCheckBox("Mirror Z")
        options_layout.addWidget(self.mirror_z)
        
        options_layout.addWidget(QLabel(""))  # Spacer
        
        self.attach_to_surface = QCheckBox("Attach to Surface")
        self.attach_to_surface.setChecked(True)
        options_layout.addWidget(self.attach_to_surface)

        # Axis anchor selection for axis-lock behavior
        anchor_layout = QHBoxLayout()
        anchor_layout.addWidget(QLabel("Axis Anchor:"))
        self.axis_anchor_combo = QComboBox()
        self.axis_anchor_combo.addItems(["Cursor", "Surface/Placement", "World Center"])
        self.axis_anchor_combo.setCurrentIndex(0)
        anchor_layout.addWidget(self.axis_anchor_combo)
        options_layout.addLayout(anchor_layout)
        
        layout.addWidget(options_group)
        
        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        self.btn_fill_box = QPushButton("Fill Selection")
        actions_layout.addWidget(self.btn_fill_box)
        
        self.btn_clear_box = QPushButton("Clear Selection")
        actions_layout.addWidget(self.btn_clear_box)
        
        self.btn_select_all = QPushButton("Select All")
        actions_layout.addWidget(self.btn_select_all)
        
        self.btn_select_none = QPushButton("Select None")
        actions_layout.addWidget(self.btn_select_none)
        
        layout.addWidget(actions_group)
        
        # Spacer
        layout.addStretch()
    
    def _on_tool_clicked(self, tool: Tool):
        """Handle tool button click."""
        self.current_tool = tool
        self.toolChanged.emit(tool)
    
    def _on_color_index_changed(self, index: int):
        """Handle color index change."""
        self.current_color = index
        # Color preview will be updated by set_current_color
    
    def set_current_tool(self, tool: Tool):
        """Set the currently selected tool."""
        self.current_tool = tool
        
        for btn in self.tool_buttons.buttons():
            if isinstance(btn, ToolButton) and btn.tool == tool:
                btn.setChecked(True)
                break
    
    def set_current_color(self, color_index: int, color: QColor = None):
        """Set the current selected color."""
        self.current_color = color_index
        self.color_spinbox.blockSignals(True)
        self.color_spinbox.setValue(color_index)
        self.color_spinbox.blockSignals(False)
        
        if color:
            self.color_preview.set_color(color, color_index)
    
    def get_current_tool(self) -> Tool:
        """Get the currently selected tool."""
        return self.current_tool
    
    def get_current_color(self) -> int:
        """Get the currently selected color index."""
        return self.current_color
    
    def get_brush_size(self) -> int:
        """Get the current brush size."""
        return self.brush_size.value()
    
    def get_brush_shape(self) -> str:
        """Get the current brush shape."""
        return self.brush_shape.currentText().lower()
    
    def get_mirror_axes(self) -> tuple:
        """Get mirror state for each axis."""
        return (
            self.mirror_x.isChecked(),
            self.mirror_y.isChecked(),
            self.mirror_z.isChecked()
        )

    def get_axis_anchor_mode(self) -> str:
        """Return axis anchor mode: 'cursor', 'surface', or 'center'. Default 'cursor'."""
        text = self.axis_anchor_combo.currentText().lower()
        if 'surface' in text:
            return 'surface'
        if 'center' in text:
            return 'center'
        return 'cursor'
