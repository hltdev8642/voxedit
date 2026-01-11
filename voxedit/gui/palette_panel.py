"""
PalettePanel - Color Palette Panel
==================================

Dock widget panel for color palette management and selection.
"""

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QScrollArea, QPushButton, QLabel, QColorDialog,
    QFileDialog, QGroupBox, QSpinBox, QFrame,
    QSizePolicy, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QColor, QPainter, QMouseEvent, QPalette

from voxedit.core.palette import VoxelPalette


class ColorSwatch(QWidget):
    """Individual color swatch widget."""
    
    clicked = pyqtSignal(int)  # color index
    rightClicked = pyqtSignal(int)  # color index
    
    def __init__(self, index: int, color: QColor, parent=None):
        super().__init__(parent)
        
        self.index = index
        self.color = color
        self.selected = False
        
        self.setFixedSize(20, 20)
        self.setToolTip(f"Color {index}\nRGB: ({color.red()}, {color.green()}, {color.blue()})")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
    
    def set_color(self, color: QColor):
        """Set the swatch color."""
        self.color = color
        self.setToolTip(f"Color {self.index}\nRGB: ({color.red()}, {color.green()}, {color.blue()})")
        self.update()
    
    def set_selected(self, selected: bool):
        """Set selection state."""
        self.selected = selected
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        
        # Draw color
        painter.fillRect(0, 0, self.width(), self.height(), self.color)
        
        # Draw selection border
        if self.selected:
            painter.setPen(QColor(255, 255, 255))
            painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
            painter.setPen(QColor(0, 0, 0))
            painter.drawRect(1, 1, self.width() - 3, self.height() - 3)
        else:
            painter.setPen(QColor(80, 80, 80))
            painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)
        elif event.button() == Qt.MouseButton.RightButton:
            self.rightClicked.emit(self.index)


class PaletteGrid(QWidget):
    """Grid of color swatches."""
    
    colorSelected = pyqtSignal(int)
    colorRightClicked = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.palette: Optional[VoxelPalette] = None
        self.swatches = []
        self.selected_index = 1
        self.columns = 16
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the palette grid UI."""
        self.layout = QGridLayout(self)
        self.layout.setSpacing(1)
        self.layout.setContentsMargins(4, 4, 4, 4)
        
        # Create 256 swatches (index 0 is empty/transparent)
        for i in range(256):
            swatch = ColorSwatch(i, QColor(0, 0, 0))
            swatch.clicked.connect(self._on_swatch_clicked)
            swatch.rightClicked.connect(self._on_swatch_right_clicked)
            
            row = i // self.columns
            col = i % self.columns
            
            self.layout.addWidget(swatch, row, col)
            self.swatches.append(swatch)
        
        # Set index 0 swatch appearance (empty)
        self.swatches[0].setToolTip("Color 0 (Empty/Air)")
        
        # Select first non-empty color
        self.swatches[1].set_selected(True)
    
    def set_palette(self, palette: VoxelPalette):
        """Set the color palette."""
        self.palette = palette
        
        for i, swatch in enumerate(self.swatches):
            color = palette.get_color(i)
            qcolor = QColor(color.r, color.g, color.b, color.a)
            swatch.set_color(qcolor)
    
    def _on_swatch_clicked(self, index: int):
        """Handle swatch click."""
        # Update selection
        self.swatches[self.selected_index].set_selected(False)
        self.swatches[index].set_selected(True)
        self.selected_index = index
        
        self.colorSelected.emit(index)
    
    def _on_swatch_right_clicked(self, index: int):
        """Handle swatch right-click."""
        self.colorRightClicked.emit(index)
    
    def get_selected_index(self) -> int:
        """Get the currently selected color index."""
        return self.selected_index
    
    def select_index(self, index: int):
        """Select a color by index."""
        if 0 <= index < 256:
            self._on_swatch_clicked(index)


class PalettePanel(QWidget):
    """
    Panel for color palette display and management.
    
    Signals:
        colorSelected: Emitted when a color is selected (index)
        paletteChanged: Emitted when the palette is modified
    """
    
    colorSelected = pyqtSignal(int)
    paletteChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.palette: Optional[VoxelPalette] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Palette grid in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.palette_grid = PaletteGrid()
        self.palette_grid.colorSelected.connect(self._on_color_selected)
        self.palette_grid.colorRightClicked.connect(self._on_color_right_clicked)
        scroll.setWidget(self.palette_grid)
        
        layout.addWidget(scroll)
        
        # Selected color info
        info_group = QGroupBox("Selected Color")
        info_layout = QVBoxLayout(info_group)
        
        # Color preview
        preview_layout = QHBoxLayout()
        
        self.color_preview = QFrame()
        self.color_preview.setFixedSize(48, 48)
        self.color_preview.setAutoFillBackground(True)
        self.color_preview.setStyleSheet("background-color: white; border: 1px solid #555;")
        preview_layout.addWidget(self.color_preview)
        
        # RGB values
        rgb_layout = QVBoxLayout()
        
        r_layout = QHBoxLayout()
        r_layout.addWidget(QLabel("R:"))
        self.spin_r = QSpinBox()
        self.spin_r.setRange(0, 255)
        self.spin_r.valueChanged.connect(self._on_rgb_changed)
        r_layout.addWidget(self.spin_r)
        rgb_layout.addLayout(r_layout)
        
        g_layout = QHBoxLayout()
        g_layout.addWidget(QLabel("G:"))
        self.spin_g = QSpinBox()
        self.spin_g.setRange(0, 255)
        self.spin_g.valueChanged.connect(self._on_rgb_changed)
        g_layout.addWidget(self.spin_g)
        rgb_layout.addLayout(g_layout)
        
        b_layout = QHBoxLayout()
        b_layout.addWidget(QLabel("B:"))
        self.spin_b = QSpinBox()
        self.spin_b.setRange(0, 255)
        self.spin_b.valueChanged.connect(self._on_rgb_changed)
        b_layout.addWidget(self.spin_b)
        rgb_layout.addLayout(b_layout)
        
        preview_layout.addLayout(rgb_layout)
        preview_layout.addStretch()
        
        info_layout.addLayout(preview_layout)
        
        # Edit color button
        self.btn_edit_color = QPushButton("Edit Color...")
        self.btn_edit_color.clicked.connect(self._edit_current_color)
        info_layout.addWidget(self.btn_edit_color)
        
        layout.addWidget(info_group)
        
        # Palette management
        manage_group = QGroupBox("Palette")
        manage_layout = QVBoxLayout(manage_group)
        
        btn_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("Load")
        self.btn_load.clicked.connect(self._load_palette)
        btn_layout.addWidget(self.btn_load)
        
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._save_palette)
        btn_layout.addWidget(self.btn_save)
        
        manage_layout.addLayout(btn_layout)
        
        btn_layout2 = QHBoxLayout()
        
        self.btn_default = QPushButton("Default")
        self.btn_default.clicked.connect(self._reset_palette)
        btn_layout2.addWidget(self.btn_default)
        
        self.btn_sort = QPushButton("Sort")
        self.btn_sort.clicked.connect(self._sort_palette)
        btn_layout2.addWidget(self.btn_sort)
        
        manage_layout.addLayout(btn_layout2)
        
        layout.addWidget(manage_group)
        
        layout.addStretch()
    
    def set_palette(self, palette: VoxelPalette):
        """Set the color palette to display."""
        self.palette = palette
        self.palette_grid.set_palette(palette)
        
        # Update selected color display
        index = self.palette_grid.get_selected_index()
        self._update_color_display(index)
    
    def _on_color_selected(self, index: int):
        """Handle color selection."""
        self._update_color_display(index)
        self.colorSelected.emit(index)
    
    def _on_color_right_clicked(self, index: int):
        """Handle color right-click (context menu)."""
        menu = QMenu(self)
        
        edit_action = menu.addAction("Edit Color...")
        edit_action.triggered.connect(lambda: self._edit_color(index))
        
        copy_action = menu.addAction("Copy Color")
        copy_action.triggered.connect(lambda: self._copy_color(index))
        
        paste_action = menu.addAction("Paste Color")
        paste_action.triggered.connect(lambda: self._paste_color(index))
        
        menu.addSeparator()
        
        select_action = menu.addAction("Select All of This Color")
        
        menu.exec(self.mapToGlobal(self.palette_grid.swatches[index].pos()))
    
    def _update_color_display(self, index: int):
        """Update the selected color display."""
        if self.palette is None:
            return
        
        color = self.palette.get_color(index)
        
        # Update preview
        self.color_preview.setStyleSheet(
            f"background-color: rgb({color.r}, {color.g}, {color.b}); "
            f"border: 1px solid #555;"
        )
        
        # Update spinboxes
        self.spin_r.blockSignals(True)
        self.spin_g.blockSignals(True)
        self.spin_b.blockSignals(True)
        
        self.spin_r.setValue(color.r)
        self.spin_g.setValue(color.g)
        self.spin_b.setValue(color.b)
        
        self.spin_r.blockSignals(False)
        self.spin_g.blockSignals(False)
        self.spin_b.blockSignals(False)
    
    def _on_rgb_changed(self):
        """Handle RGB spinbox value change."""
        if self.palette is None:
            return
        
        index = self.palette_grid.get_selected_index()
        r = self.spin_r.value()
        g = self.spin_g.value()
        b = self.spin_b.value()
        
        self.palette.set_color(index, r, g, b)
        
        # Update swatch
        self.palette_grid.swatches[index].set_color(QColor(r, g, b))
        
        # Update preview
        self.color_preview.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); "
            f"border: 1px solid #555;"
        )
        
        self.paletteChanged.emit()
    
    def _edit_current_color(self):
        """Edit the currently selected color."""
        index = self.palette_grid.get_selected_index()
        self._edit_color(index)
    
    def _edit_color(self, index: int):
        """Open color dialog for editing a color."""
        if self.palette is None:
            return
        
        color = self.palette.get_color(index)
        initial = QColor(color.r, color.g, color.b, color.a)
        
        new_color = QColorDialog.getColor(
            initial, self, f"Edit Color {index}",
            QColorDialog.ColorDialogOption.ShowAlphaChannel
        )
        
        if new_color.isValid():
            self.palette.set_color(
                index,
                new_color.red(),
                new_color.green(),
                new_color.blue(),
                new_color.alpha()
            )
            
            self.palette_grid.swatches[index].set_color(new_color)
            
            if index == self.palette_grid.get_selected_index():
                self._update_color_display(index)
            
            self.paletteChanged.emit()
    
    def _copy_color(self, index: int):
        """Copy a color to clipboard (internal)."""
        # Store for internal paste
        self._copied_color = self.palette.get_color(index)
    
    def _paste_color(self, index: int):
        """Paste copied color to index."""
        if hasattr(self, '_copied_color') and self._copied_color:
            c = self._copied_color
            self.palette.set_color(index, c.r, c.g, c.b, c.a)
            self.palette_grid.swatches[index].set_color(QColor(c.r, c.g, c.b, c.a))
            
            if index == self.palette_grid.get_selected_index():
                self._update_color_display(index)
            
            self.paletteChanged.emit()
    
    def _load_palette(self):
        """Load palette from file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Palette",
            "", "Palette Files (*.pal *.vox *.png);;All Files (*)"
        )
        
        if filepath and self.palette:
            try:
                self.palette.load_from_file(filepath)
                self.palette_grid.set_palette(self.palette)
                self._update_color_display(self.palette_grid.get_selected_index())
                self.paletteChanged.emit()
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self, "Load Error",
                    f"Failed to load palette:\n{str(e)}"
                )
    
    def _save_palette(self):
        """Save palette to file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Palette",
            "", "Palette Files (*.pal);;PNG Image (*.png);;All Files (*)"
        )
        
        if filepath and self.palette:
            try:
                self.palette.save_to_file(filepath)
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self, "Save Error",
                    f"Failed to save palette:\n{str(e)}"
                )
    
    def _reset_palette(self):
        """Reset to default MagicaVoxel palette."""
        if self.palette:
            from PyQt6.QtWidgets import QMessageBox
            
            result = QMessageBox.question(
                self, "Reset Palette",
                "Reset to default MagicaVoxel palette?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if result == QMessageBox.StandardButton.Yes:
                self.palette.load_default()
                self.palette_grid.set_palette(self.palette)
                self._update_color_display(self.palette_grid.get_selected_index())
                self.paletteChanged.emit()
    
    def _sort_palette(self):
        """Sort palette colors by hue."""
        if self.palette:
            self.palette.sort_by_hue()
            self.palette_grid.set_palette(self.palette)
            self._update_color_display(self.palette_grid.get_selected_index())
            self.paletteChanged.emit()
    
    def get_selected_color_index(self) -> int:
        """Get the currently selected color index."""
        return self.palette_grid.get_selected_index()
    
    def select_color(self, index: int):
        """Select a color by index."""
        self.palette_grid.select_index(index)
