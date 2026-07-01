import sys
import math
import uuid
from qtpy.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QGraphicsItem, QGraphicsView, 
                            QGraphicsScene, QFrame, QPushButton, QListWidget, QFileDialog)
from qtpy.QtGui import QColor, QPen, QBrush, QPainter, QPainterPath, QPainterPathStroker, QPixmap, QFont, QMouseEvent
from qtpy.QtCore import Qt, QRectF, QPointF, Signal as pyqtSignal


class BaseSystemHandle(QGraphicsItem):
    """
    Abstract structural base class for all interactive manipulation nodes.
    Handles basic hover tracking, press caching, and structural bounds.
    """
    def __init__(self, color: QColor, size: float, interaction_margin: float = 10.0, parent=None):
        super().__init__(parent)
        self.color = color
        self.size = size
        self.interaction_margin = interaction_margin
        self._hovered = False
        
        self.setAcceptHoverEvents(True)
        self.setFlags(QGraphicsItem.ItemSendsGeometryChanges)
        
        # Drag caching vectors
        self._drag_start_scene = QPointF()
        self._drag_start_parent_pos = QPointF()
        self._drag_start_local_mouse = QPointF()
        self._drag_start_bbox_center = QPointF()

    def boundingRect(self):
        s = self.size + self.interaction_margin
        return QRectF(-s, -s, s * 2, s * 2)

    def shape(self):
        path = QPainterPath()
        s = self.size + self.interaction_margin
        path.addEllipse(QRectF(-s, -s, s * 2, s * 2))
        return path

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.unsetCursor()
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start_scene = event.scenePos()
            parent = self.parentItem()
            if parent:
                self._drag_start_parent_pos = parent.pos()
                self._drag_start_bbox_center = QPointF(parent._bbox_cx, parent._bbox_cy)
                self._drag_start_local_mouse = parent.mapFromScene(event.scenePos())
            event.accept()
        else:
            super().mousePressEvent(event)

class OriginHandle(BaseSystemHandle):
    """Handle that controls full 2D coordinate system translation."""
    def __init__(self, color: QColor, parent=None):
        super().__init__(color, size=14, interaction_margin=10, parent=parent)

    def hoverEnterEvent(self, event):
        self.setCursor(Qt.CrossCursor)
        super().hoverEnterEvent(event)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)
        fill = self.color.lighter(130) if self._hovered else self.color
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(fill))
        
        r = self.size / 2
        painter.drawEllipse(QPointF(0, 0), r, r)
        
        alpha = 140 if self._hovered else 90
        painter.setPen(QPen(QColor(0, 0, 0, alpha), 1.2))
        painter.drawLine(QPointF(-r * 1.5, 0), QPointF(r * 1.5, 0))
        painter.drawLine(QPointF(0, -r * 1.5), QPointF(0, r * 1.5))

    def mouseMoveEvent(self, event):
        parent = self.parentItem()
        if parent and event.buttons() & Qt.LeftButton:
            delta = event.scenePos() - self._drag_start_scene
            parent.setPos(self._drag_start_parent_pos + delta)
            parent.parent_widget.notify_system_changed(parent)
            event.accept()


class AxisHandle(BaseSystemHandle):
    """Handle that controls coordinate system vector orientation."""
    def __init__(self, color: QColor, parent=None):
        super().__init__(color, size=10, interaction_margin=10, parent=parent)

    def hoverEnterEvent(self, event):
        self.setCursor(Qt.SizeAllCursor)
        super().hoverEnterEvent(event)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)
        fill = self.color.lighter(130) if self._hovered else self.color
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(fill))
        painter.drawEllipse(QPointF(0, 0), self.size / 2, self.size / 2)

    def mouseMoveEvent(self, event):
        parent = self.parentItem()
        if parent and event.buttons() & Qt.LeftButton:
            local_mouse_pos = parent.mapFromScene(event.scenePos())
            parent.handle_axis_drag(self, local_mouse_pos)
            event.accept()


class BBoxCornerHandle(BaseSystemHandle):
    """Handle that controls localized resizing of a specific bounding box corner."""
    def __init__(self, color: QColor, corner_id: str, parent=None):
        super().__init__(color, size=8, interaction_margin=10, parent=parent)
        self.corner_id = corner_id

    def hoverEnterEvent(self, event):
        if self.corner_id in ["tl", "br"]:
            self.setCursor(Qt.SizeFDiagCursor)
        else:
            self.setCursor(Qt.SizeBDiagCursor)
        super().hoverEnterEvent(event)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)
        fill = self.color.lighter(130) if self._hovered else self.color
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(fill))
        r = self.size / 2
        painter.drawRect(QRectF(-r, -r, self.size, self.size))

    def mouseMoveEvent(self, event):
        parent = self.parentItem()
        if parent and event.buttons() & Qt.LeftButton:
            local_mouse_pos = parent.mapFromScene(event.scenePos())
            parent.handle_bbox_resize(self, local_mouse_pos)
            event.accept()

class InteractiveCoordinateSystem(QGraphicsItem):
    """
    Composite graphics framework item composing an Origin, an X-Axis, a Y-Axis, 
    and a translation-decoupled bounding box with four-corner boundary clamping.
    """
    def __init__(self, sys_id: str, sys_index: int, initial_pos: QPointF, parent_widget):
        super().__init__()
        self.sys_id = sys_id
        self._sys_index = sys_index  
        self.parent_widget = parent_widget
        
        self.setFlags(QGraphicsItem.ItemIsSelectable)
        self.setPos(initial_pos)

        self._len_x = 75.0
        self._len_y = 150.0
        self._current_angle = 0.0

        self._bbox_w = 200.0
        self._bbox_h = 200.0
        self._bbox_cx = 0.0
        self._bbox_cy = 0.0

        self.color_origin = QColor(255, 255, 255, 120)
        self.color_x = QColor(230, 159, 0)        
        self.color_y = QColor(86, 180, 233)       
        self.color_bbox = QColor(255, 255, 255, 160)

        # Inherited Component Subclasses
        self.origin = OriginHandle(self.color_origin, parent=self)
        self.origin.setPos(0, 0)

        self.axis_x = AxisHandle(self.color_x, parent=self)
        self.axis_y = AxisHandle(self.color_y, parent=self)
        
        self.bbox_tl = BBoxCornerHandle(self.color_bbox, "tl", parent=self)
        self.bbox_tr = BBoxCornerHandle(self.color_bbox, "tr", parent=self)
        self.bbox_bl = BBoxCornerHandle(self.color_bbox, "bl", parent=self)
        self.bbox_br = BBoxCornerHandle(self.color_bbox, "br", parent=self)

        self.update_axis_positions()
        self.update_bbox_positions()

    def update_axis_positions(self):
        x_target = QPointF(math.cos(self._current_angle) * self._len_x, 
                           math.sin(self._current_angle) * self._len_x)
        self.axis_x.setPos(x_target)
        
        y_angle = self._current_angle - (math.pi / 2.0)
        y_target = QPointF(math.cos(y_angle) * self._len_y, 
                           math.sin(y_angle) * self._len_y)
        self.axis_y.setPos(y_target)

    def update_bbox_positions(self):
        hw = self._bbox_w / 2.0
        hh = self._bbox_h / 2.0
        
        self.bbox_tl.setPos(self._bbox_cx - hw, self._bbox_cy - hh)
        self.bbox_tr.setPos(self._bbox_cx + hw, self._bbox_cy - hh)
        self.bbox_bl.setPos(self._bbox_cx - hw, self._bbox_cy + hh)
        self.bbox_br.setPos(self._bbox_cx + hw, self._bbox_cy + hh)

    def get_bbox_rect(self):
        return QRectF(self._bbox_cx - self._bbox_w / 2.0, 
                      self._bbox_cy - self._bbox_h / 2.0, 
                      self._bbox_w, self._bbox_h)

    def boundingRect(self):
        bbox = self.get_bbox_rect()
        axes_rect = QRectF(self.origin.pos(), self.axis_x.pos()).united(QRectF(self.origin.pos(), self.axis_y.pos()))
        return bbox.united(axes_rect).adjusted(-35, -20, 20, 35)

    def shape(self):
        path = QPainterPath()
        path.moveTo(self.origin.pos())
        path.lineTo(self.axis_x.pos())
        path.moveTo(self.origin.pos())
        path.lineTo(self.axis_y.pos())
        path.addRect(self.get_bbox_rect())
        
        stroker = QPainterPathStroker()
        stroker.setWidth(16)
        stroker.setCapStyle(Qt.RoundCap)
        
        total_shape = stroker.createStroke(path)
        total_shape.addPath(self.origin.mapToParent(self.origin.shape()))
        total_shape.addPath(self.axis_x.mapToParent(self.axis_x.shape()))
        total_shape.addPath(self.axis_y.mapToParent(self.axis_y.shape()))
        total_shape.addPath(self.bbox_tl.mapToParent(self.bbox_tl.shape()))
        total_shape.addPath(self.bbox_tr.mapToParent(self.bbox_tr.shape()))
        total_shape.addPath(self.bbox_bl.mapToParent(self.bbox_bl.shape()))
        total_shape.addPath(self.bbox_br.mapToParent(self.bbox_br.shape()))
        
        return total_shape

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.setPen(QPen(self.color_bbox, 2.0, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(self.get_bbox_rect())

        painter.setPen(QPen(QColor(self.color_x.red(), self.color_x.green(), self.color_x.blue(), 230), 3.0))
        painter.drawLine(self.origin.pos(), self.axis_x.pos())
        
        painter.setPen(QPen(QColor(self.color_y.red(), self.color_y.green(), self.color_y.blue(), 230), 3.0))
        painter.drawLine(self.origin.pos(), self.axis_y.pos())

        # Index Label (High contrast dropping double outline)
        label = str(self._sys_index)
        font = QFont("Arial", 12, QFont.Bold)
        painter.setFont(font)
        
        painter.setPen(QColor(0, 0, 0, 255))
        painter.drawText(QRectF(-23, 11, 20, 20), Qt.AlignRight | Qt.AlignTop, label)
        painter.drawText(QRectF(-25, 9, 20, 20), Qt.AlignRight | Qt.AlignTop, label)

        painter.setPen(QColor(255, 255, 255, 240))
        painter.drawText(QRectF(-24, 10, 20, 20), Qt.AlignRight | Qt.AlignTop, label)

    def handle_axis_drag(self, node, local_mouse_pos):
        self.prepareGeometryChange()
        dx, dy = local_mouse_pos.x(), local_mouse_pos.y()
        
        if node == self.axis_x:
            self._current_angle = math.atan2(dy, dx)
        elif node == self.axis_y:
            self._current_angle = math.atan2(dy, dx) + (math.pi / 2.0)

        self.update_axis_positions()
        self.update()
        self.parent_widget.notify_system_changed(self)

    def handle_bbox_translate(self, target_cx, target_cy):
        self.prepareGeometryChange()
        hw, hh = self._bbox_w / 2.0, self._bbox_h / 2.0
        self._bbox_cx = max(-hw, min(hw, target_cx))
        self._bbox_cy = max(-hh, min(hh, target_cy))
        self.update_bbox_positions()
        self.update()
        self.parent_widget.notify_system_changed(self)

    def handle_bbox_resize(self, node, local_mouse_pos):
        self.prepareGeometryChange()
        hw, hh = self._bbox_w / 2.0, self._bbox_h / 2.0
        x1, x2 = self._bbox_cx - hw, self._bbox_cx + hw  
        y1, y2 = self._bbox_cy - hh, self._bbox_cy + hh  

        if "br" in node.corner_id:
            x2, y2 = max(0.0, local_mouse_pos.x()), max(0.0, local_mouse_pos.y())
        elif "tl" in node.corner_id:
            x1, y1 = min(0.0, local_mouse_pos.x()), min(0.0, local_mouse_pos.y())
        elif "tr" in node.corner_id:
            x2, y1 = max(0.0, local_mouse_pos.x()), min(0.0, local_mouse_pos.y())
        elif "bl" in node.corner_id:
            x1, y2 = min(0.0, local_mouse_pos.x()), max(0.0, local_mouse_pos.y())

        self._bbox_w = max(20.0, x2 - x1)
        self._bbox_h = max(20.0, y2 - y1)
        self._bbox_cx = x1 + self._bbox_w / 2.0
        self._bbox_cy = y1 + self._bbox_h / 2.0

        self.update_bbox_positions()
        self.update()
        self.parent_widget.notify_system_changed(self)

    def get_data(self):
        origin_scene_pos = self.mapToScene(self.origin.pos())
        dy = self.axis_y.pos().y() - self.origin.pos().y()
        dx = self.axis_y.pos().x() - self.origin.pos().x()
        angle_deg = math.degrees(math.atan2(dy, dx))
        
        return {
            "id": self.sys_id,
            "index": self._sys_index,
            "origin_x": origin_scene_pos.x(),
            "origin_y": origin_scene_pos.y(),
            "y_axis_angle_deg": angle_deg,
            "scale_x_len": self._len_x,
            "scale_y_len": self._len_y,
            "bbox_width": self._bbox_w,
            "bbox_height": self._bbox_h,
            "bbox_offset_x": self._bbox_cx,
            "bbox_offset_y": self._bbox_cy
        }


class MultiCoordViewer(QGraphicsView):
    system_updated = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor(40, 40, 40)))
        
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        self.coordinate_systems = {}
        self.bg_pixmap_item = None

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor
        scale_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.scale(scale_factor, scale_factor)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        """Intercept middle mouse clicks to activate canvas drag pan mode."""
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            # Safe floating-point QPointF translation mapping compatible with Qt 6 signatures
            fake_event = QMouseEvent(
                event.type(), event.position(), Qt.LeftButton, 
                event.buttons() | Qt.LeftButton, event.modifiers()
            )
            super().mousePressEvent(fake_event)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Restore workspace selection tools when releasing pan mode."""
        if event.button() == Qt.MiddleButton:
            fake_event = QMouseEvent(
                event.type(), event.position(), Qt.LeftButton, 
                event.buttons() & ~Qt.LeftButton, event.modifiers()
            )
            super().mouseReleaseEvent(fake_event)
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            super().mouseReleaseEvent(event)

    def load_background_image(self, file_path: str):
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            return False
        if self.bg_pixmap_item in self.scene.items():
            self.scene.removeItem(self.bg_pixmap_item)
            
        self.bg_pixmap_item = self.scene.addPixmap(pixmap)
        self.bg_pixmap_item.setZValue(-100)
        self.setSceneRect(QRectF(pixmap.rect()))
        return True

    def add_coordinate_system(self, name: str, sys_index: int, scene_pos: QPointF):
        sys_id = f"{name} [#{sys_index}] ({str(uuid.uuid4())[:4]})"
        coord_sys = InteractiveCoordinateSystem(sys_id, sys_index, scene_pos, self)
        self.scene.addItem(coord_sys)
        self.coordinate_systems[sys_id] = coord_sys
        self.notify_system_changed(coord_sys)
        return sys_id

    def remove_coordinate_system(self, sys_id: str):
        if sys_id in self.coordinate_systems:
            item = self.coordinate_systems.pop(sys_id)
            self.scene.removeItem(item)
            self.scene.update()

    def notify_system_changed(self, system: InteractiveCoordinateSystem):
        self.system_updated.emit(system.get_data())


class MainApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Refactored Modular Image Workspace Navigation")
        self.resize(1200, 750)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        self.viewer = MultiCoordViewer()
        self.viewer.setSceneRect(0, 0, 1000, 1000)
        self.viewer.system_updated.connect(self.update_telemetry_display)
        main_layout.addWidget(self.viewer, stretch=3)
        
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel, stretch=1)
        
        self.load_img_btn = QPushButton("📷 Load Background Image")
        self.load_img_btn.setStyleSheet("font-weight: bold; background-color: #2b5c8f; color: white; padding: 6px;")
        self.load_img_btn.clicked.connect(self.open_image_dialog)
        control_layout.addWidget(self.load_img_btn)
        
        control_layout.addSpacing(10)
        
        self.add_btn = QPushButton("Add Coordinate System")
        self.add_btn.clicked.connect(self.spawn_system)
        control_layout.addWidget(self.add_btn)
        
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self.delete_selected_system)
        control_layout.addWidget(self.remove_btn)
        
        control_layout.addWidget(QLabel("Systems Active Registry:"))
        self.system_list = QListWidget()
        control_layout.addWidget(self.system_list)
        
        control_layout.addWidget(QLabel("Live System Metrics:"))
        self.telemetry_label = QLabel("Interact with or select a system map.")
        self.telemetry_label.setWordWrap(True)
        self.telemetry_label.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.telemetry_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; padding: 8px; background: #1e1e1e; color: #56B4E9;"
        )
        control_layout.addWidget(self.telemetry_label)

        self.spawn_system()

    def open_image_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Background Image Asset", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.viewer.load_background_image(file_path)

    def spawn_system(self):
        idx = self.system_list.count()
        center_view = self.viewer.viewport().rect().center()
        scene_pt = self.viewer.mapToScene(center_view)
        scene_pt += QPointF(idx * 45, idx * 45)
        
        sys_id = self.viewer.add_coordinate_system(f"System_{idx + 1}", idx, scene_pt)
        self.system_list.addItem(sys_id)

    def delete_selected_system(self):
        current_item = self.system_list.currentItem()
        if current_item:
            sys_id = current_item.text()
            self.viewer.remove_coordinate_system(sys_id)
            self.system_list.takeItem(self.system_list.row(current_item))
            self.telemetry_label.setText("System cleared.")

    def update_telemetry_display(self, data: dict):
        text = (
            f"SYSTEM ID   : {data['id']}\n"
            f"INDEX MAP   : {data['index']}\n"
            f"---------------------------\n"
            f"Origin X    : {data['origin_x']:.2f} px\n"
            f"Origin Y    : {data['origin_y']:.2f} px\n"
            f"Y-Axis ∠    : {data['y_axis_angle_deg']:.2f}°\n"
            f"X length    : {data['scale_x_len']:.1f} px\n"
            f"Y length    : {data['scale_y_len']:.1f} px\n"
            f"BBox Size   : {data['bbox_width']:.1f} x {data['bbox_height']:.1f} px\n"
            f"BBox Offset : [{data['bbox_offset_x']:.1f}, {data['bbox_offset_y']:.1f}] px\n"
            f"Navigation  : Wheel = Zoom | Middle-Click = Pan"
        )
        self.telemetry_label.setText(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApplication()
    win.show()
    sys.exit(app.exec_())