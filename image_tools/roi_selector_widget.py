from PyQt5.QtWidgets import QDialog, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QListWidget, QPushButton
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen
from PyQt5.QtCore import Qt, QRect
from qt_widgets import NDarray_to_QPixmap

class ROISelectorWidget(QWidget):

    def __init__(self, image = None, *args, **kwargs):
    
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
        # redraw on top of image
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

    def on_delete(self):
        if len(self.ROIs) > 0:
            idx = self.roi_list.currentRow()
            self.roi_list.takeItem(idx)
            self.ROIs.pop(idx)

    def on_done(self):
        pass

    def on_roi_selection(self, index):
        self.current_roi = self.ROIs[index]

    def on_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            # left-click always create a new ROI
            pos = event.pos()
            self.current_roi = QRect(pos.x(),pos.y(),0,0)
            self.ROIs.append(self.current_roi)
            idx = len(self.ROIs)
            self.roi_list.addItem(str(idx-1))
            self.roi_list.setCurrentRow(idx-1)
        elif event.button() == Qt.RightButton:
            # you can resize current ROI with right-click
            pass
        self.update()

    def on_mouse_move(self, event):
        pos = event.pos() 
        self.ROIs[-1].setBottomRight(pos)
        self.current_roi = self.ROIs[-1]
        self.update()

    def get_ROIs(self):
        return [(r.x(), r.y(), r.width(), r.height()) for r in self.ROIs]


class ROISelectorDialog(QDialog,ROISelectorWidget):
    def on_done(self):
        self.accept()