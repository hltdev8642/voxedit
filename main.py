#!/usr/bin/env python3
"""
VoxEdit - Teardown Voxel Editor
===============================

Main entry point for the VoxEdit application.
A powerful voxel editor with support for multiple formats including
MagicaVoxel (.vox), Minecraft schematics, and mesh exports.

Usage:
    python main.py [file]
    
Arguments:
    file    Optional voxel file to open on startup
"""

import sys
import os
import argparse
from pathlib import Path

# Add the project root to path if needed
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def setup_high_dpi():
    """Configure high DPI settings before QApplication creation."""
    os.environ.setdefault('QT_ENABLE_HIGHDPI_SCALING', '1')
    os.environ.setdefault('QT_SCALE_FACTOR_ROUNDING_POLICY', 'PassThrough')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='VoxEdit - Teardown Voxel Editor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported formats:
  Import: .vox, .schematic, .schem, .litematic, .binvox, .qb, .kvx
  Export: .vox, .obj, .stl, .ply, .gltf, .schematic

Examples:
  %(prog)s                     Start with empty model
  %(prog)s model.vox           Open a MagicaVoxel file
  %(prog)s build.schematic     Open a Minecraft schematic
        """
    )
    
    parser.add_argument(
        'file',
        nargs='?',
        help='Voxel file to open'
    )
    
    parser.add_argument(
        '--size',
        type=str,
        default='32x32x32',
        help='Initial model size for new files (default: 32x32x32)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    return parser.parse_args()


def main():
    """Main application entry point."""
    # Setup before QApplication
    setup_high_dpi()
    
    # Parse arguments
    args = parse_arguments()
    
    # Import Qt after high DPI setup
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QIcon, QPalette, QColor
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("VoxEdit")
    app.setOrganizationName("VoxEdit")
    app.setApplicationVersion("1.0.0")
    
    # Set dark theme
    set_dark_theme(app)
    
    # Import main window
    from voxedit.gui import MainWindow
    
    # Create and show main window
    window = MainWindow()
    
    # Open file if provided
    if args.file:
        filepath = Path(args.file)
        if filepath.exists():
            window.open_file(str(filepath))
        else:
            print(f"Warning: File not found: {args.file}")
    else:
        # Parse initial size
        try:
            parts = args.size.lower().split('x')
            size = (int(parts[0]), int(parts[1]), int(parts[2]))
            window.new_model(size)
        except (ValueError, IndexError):
            window.new_model((32, 32, 32))
    
    window.show()
    
    # Run application
    return app.exec()


def set_dark_theme(app: 'QApplication'):
    """Apply a dark theme to the application."""
    from PyQt6.QtGui import QPalette, QColor
    from PyQt6.QtCore import Qt
    
    palette = QPalette()
    
    # Base colors
    dark = QColor(45, 45, 48)
    darker = QColor(30, 30, 32)
    darkest = QColor(20, 20, 22)
    
    light = QColor(200, 200, 200)
    lighter = QColor(230, 230, 230)
    
    accent = QColor(0, 120, 215)
    accent_hover = QColor(30, 140, 235)
    
    # Set palette colors
    palette.setColor(QPalette.ColorRole.Window, dark)
    palette.setColor(QPalette.ColorRole.WindowText, light)
    palette.setColor(QPalette.ColorRole.Base, darker)
    palette.setColor(QPalette.ColorRole.AlternateBase, dark)
    palette.setColor(QPalette.ColorRole.ToolTipBase, darkest)
    palette.setColor(QPalette.ColorRole.ToolTipText, lighter)
    palette.setColor(QPalette.ColorRole.Text, light)
    palette.setColor(QPalette.ColorRole.Button, dark)
    palette.setColor(QPalette.ColorRole.ButtonText, light)
    palette.setColor(QPalette.ColorRole.BrightText, lighter)
    palette.setColor(QPalette.ColorRole.Link, accent)
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, lighter)
    
    # Disabled colors
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, 
                     QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,
                     QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText,
                     QColor(127, 127, 127))
    
    app.setPalette(palette)
    
    # Additional stylesheet for polish
    app.setStyleSheet("""
        QMainWindow {
            background-color: #2d2d30;
        }
        
        QMenuBar {
            background-color: #2d2d30;
            color: #c8c8c8;
            border-bottom: 1px solid #3f3f46;
        }
        
        QMenuBar::item:selected {
            background-color: #3e3e42;
        }
        
        QMenu {
            background-color: #2d2d30;
            color: #c8c8c8;
            border: 1px solid #3f3f46;
        }
        
        QMenu::item:selected {
            background-color: #094771;
        }
        
        QMenu::separator {
            height: 1px;
            background-color: #3f3f46;
            margin: 4px 8px;
        }
        
        QToolBar {
            background-color: #2d2d30;
            border: none;
            spacing: 4px;
            padding: 4px;
        }
        
        QToolBar::separator {
            width: 1px;
            background-color: #3f3f46;
            margin: 4px 8px;
        }
        
        QToolButton {
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 4px;
            padding: 4px;
        }
        
        QToolButton:hover {
            background-color: #3e3e42;
            border-color: #3f3f46;
        }
        
        QToolButton:pressed {
            background-color: #094771;
        }
        
        QToolButton:checked {
            background-color: #094771;
            border-color: #0078d7;
        }
        
        QDockWidget {
            titlebar-close-icon: url(close.png);
            titlebar-normal-icon: url(float.png);
        }
        
        QDockWidget::title {
            background-color: #2d2d30;
            padding: 6px;
            border-bottom: 1px solid #3f3f46;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 1px solid #3f3f46;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 8px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
        }
        
        QScrollBar:vertical {
            background-color: #2d2d30;
            width: 14px;
            margin: 0;
        }
        
        QScrollBar::handle:vertical {
            background-color: #5a5a5e;
            min-height: 30px;
            border-radius: 4px;
            margin: 2px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #6a6a6e;
        }
        
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0;
        }
        
        QScrollBar:horizontal {
            background-color: #2d2d30;
            height: 14px;
            margin: 0;
        }
        
        QScrollBar::handle:horizontal {
            background-color: #5a5a5e;
            min-width: 30px;
            border-radius: 4px;
            margin: 2px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background-color: #6a6a6e;
        }
        
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {
            width: 0;
        }
        
        QStatusBar {
            background-color: #007acc;
            color: white;
        }
        
        QStatusBar::item {
            border: none;
        }
        
        QPushButton {
            background-color: #3c3c3c;
            color: #c8c8c8;
            border: 1px solid #3f3f46;
            border-radius: 4px;
            padding: 6px 16px;
            min-width: 80px;
        }
        
        QPushButton:hover {
            background-color: #4e4e52;
            border-color: #5a5a5e;
        }
        
        QPushButton:pressed {
            background-color: #094771;
        }
        
        QPushButton:disabled {
            background-color: #2d2d30;
            color: #5a5a5e;
        }
        
        QSpinBox, QDoubleSpinBox {
            background-color: #1e1e1e;
            color: #c8c8c8;
            border: 1px solid #3f3f46;
            border-radius: 4px;
            padding: 4px;
        }
        
        QSpinBox:focus, QDoubleSpinBox:focus {
            border-color: #0078d7;
        }
        
        QComboBox {
            background-color: #3c3c3c;
            color: #c8c8c8;
            border: 1px solid #3f3f46;
            border-radius: 4px;
            padding: 4px 8px;
        }
        
        QComboBox:hover {
            border-color: #5a5a5e;
        }
        
        QComboBox::drop-down {
            border: none;
            padding-right: 8px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #2d2d30;
            color: #c8c8c8;
            border: 1px solid #3f3f46;
            selection-background-color: #094771;
        }
        
        QSlider::groove:horizontal {
            background-color: #3f3f46;
            height: 4px;
            border-radius: 2px;
        }
        
        QSlider::handle:horizontal {
            background-color: #0078d7;
            width: 16px;
            height: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }
        
        QSlider::handle:horizontal:hover {
            background-color: #1c97ea;
        }
        
        QCheckBox {
            spacing: 8px;
        }
        
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid #5a5a5e;
            border-radius: 2px;
            background-color: #1e1e1e;
        }
        
        QCheckBox::indicator:checked {
            background-color: #0078d7;
            border-color: #0078d7;
        }
        
        QCheckBox::indicator:hover {
            border-color: #0078d7;
        }
        
        QLabel {
            color: #c8c8c8;
        }
        
        QProgressBar {
            background-color: #3f3f46;
            border: none;
            border-radius: 4px;
            text-align: center;
            color: white;
        }
        
        QProgressBar::chunk {
            background-color: #0078d7;
            border-radius: 4px;
        }
    """)


if __name__ == "__main__":
    sys.exit(main())
