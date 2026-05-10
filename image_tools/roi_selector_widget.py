from qtpy.QtWidgets import (QDialog, QWidget, QLabel, QHBoxLayout, 
                            QVBoxLayout, QListWidget, QPushButton, 
                            QApplication)
from qtpy.QtGui import QPainter, QColor, QBrush, QPen
from qtpy.QtCore import Qt, QRect
from qt_widgets import NDarray_to_QPixmap

class ROISelectorWidget(QWidget):

    def __init__(self, image=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.assignment = None
        self.image = image 
        self.current_roi = QRect()
        self.ROIs = []
        self.declare_components()
        self.layout_components()
    
    def declare_components(self):
        self.image_label = QLabel(self)
        if self.image is not None:
            self.image_label.setPixmap(NDarray_to_QPixmap(self.image))
        
        self.image_label.mousePressEvent = self.on_mouse_press
        self.image_label.mouseMoveEvent = self.on_mouse_move

        self.roi_list = QListWidget(self)
        self.roi_list.currentRowChanged.connect(self.on_roi_selection)

        self.delete_roi_button = QPushButton('delete', self)
        self.delete_roi_button.clicked.connect(self.on_delete)

        self.done = QPushButton('done', self)
        self.done.clicked.connect(self.on_done)

    def layout_components(self):
        buttons = QHBoxLayout()
        buttons.addWidget(self.delete_roi_button)
        buttons.addWidget(self.done)

        right_panel = QVBoxLayout()
        right_panel.addWidget(self.roi_list)
        right_panel.addLayout(buttons)

        layout = QHBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addLayout(right_panel)

    def paintEvent(self, event):
        if self.image is not None:
            self.image_label.setPixmap(NDarray_to_QPixmap(self.image))
            painter = QPainter(self.image_label.pixmap())
            pen = QPen()
            pen.setWidth(3)
            for roi in self.ROIs:
                pen_color = QColor(70, 0, 0, 60)
                brush_color = QColor(100, 10, 10, 40) 
                if roi == self.current_roi:
                    pen_color = QColor(0, 70, 0, 60)
                    brush_color = QColor(10, 100, 10, 40) 
                pen.setColor(pen_color)
                brush = QBrush(brush_color)  
                painter.setPen(pen)
                painter.setBrush(brush)   
                painter.drawRect(roi)
            # It's good practice to end the painter explicitly
            painter.end()

    def on_delete(self):
        if len(self.ROIs) > 0:
            idx = self.roi_list.currentRow()
            if idx >= 0: # Ensure something is actually selected
                self.roi_list.takeItem(idx)
                self.ROIs.pop(idx)
                self.update()

    def on_done(self):
        pass

    def on_roi_selection(self, index):
        if 0 <= index < len(self.ROIs):
            self.current_roi = self.ROIs[index]
            self.update()

    def on_mouse_press(self, event):
        # In QtPy/PySide/PyQt6, event.button() is the standard way to check
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            self.current_roi = QRect(pos.x(), pos.y(), 0, 0)
            self.ROIs.append(self.current_roi)
            idx = len(self.ROIs)
            self.roi_list.addItem(str(idx-1))
            self.roi_list.setCurrentRow(idx-1)
        self.update()

    def on_mouse_move(self, event):
        if len(self.ROIs) > 0:
            pos = event.pos() 
            self.ROIs[-1].setBottomRight(pos)
            self.current_roi = self.ROIs[-1]
            self.update()

    def get_ROIs(self):
        return [(r.x(), r.y(), r.width(), r.height()) for r in self.ROIs]


class ROISelectorDialog(QDialog, ROISelectorWidget):
    def on_done(self):
        self.accept()

if __name__ == "__main__":
    
    import sys
    import numpy as np

    app = QApplication(sys.argv)
    test_image = np.zeros((512, 512, 3), dtype=np.uint8)
    
    test_image[:, :, 0] = 50   # Dark Red background
    test_image[100:400, 100:400, 1] = 150  # Green square in middle

    dialog = ROISelectorDialog(image=test_image)
    dialog.setWindowTitle("ROI Selector Test")

    if dialog.exec_():
        rois = dialog.get_ROIs()
        print(f"Selection complete. Captured {len(rois)} ROIs:")
        for i, rect in enumerate(rois):
            print(f" ROI {i}: x={rect[0]}, y={rect[1]}, w={rect[2]}, h={rect[3]}")
    else:
        print("User cancelled the selection.")