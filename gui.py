#!/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2  # opencv
from PySide6.QtCore import Qt, QSize, QEvent, Signal, QObject, QPointF
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPalette,
    QPainter,
    QMouseEvent,
    QWheelEvent,
    QAction,
)
from PySide6.QtPrintSupport import QPrintDialog, QPrinter
from PySide6.QtWidgets import (
    QLabel,
    QSizePolicy,
    QScrollArea,
    QMessageBox,
    QMainWindow,
    QMenu,
    #    qApp,
    QFileDialog,
    QWidget,
    QHBoxLayout,
    QTabWidget,
)
import alfr
from alfr import Shot, Renderer, Camera
from alfr.shot import load_shots_from_json


# https://stackoverflow.com/questions/41688668/how-to-return-mouse-coordinates-in-realtime
class MouseTracker(QLabel):
    rotateEvent = Signal(QPointF)
    panEvent = Signal(QPointF)
    zoomEvent = Signal(QPointF)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(
            True
        )  # mouseMoveEvent is called when the mouse moves over the label (also when no mouse button is pressed)

        self.setMinimumSize(512, 512)
        self.setMaximumSize(512, 512)

        # self.initUI()
        # self.setMouseTracking(True)

        self._lastpos = QPointF(0, 0)

        # only for testing:
        self.rotateEvent.connect(lambda dxy: print("rotateEvent", dxy))
        self.panEvent.connect(lambda dxy: print("panEvent", dxy))
        self.zoomEvent.connect(lambda dxy: print("zoomEvnent", dxy))

    @property
    def lastpos(self) -> QPointF:
        return self._lastpos

    @lastpos.setter
    def lastpos(self, value: QPointF):
        self._lastpos = value

    def initUI(self):
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle("Mouse Tracker")
        self.label = QLabel(self)
        self.label.resize(200, 40)
        self.show()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:

        currpos = event.position()
        dxy = currpos - self.lastpos

        if event.buttons() == Qt.LeftButton:
            print("Mouse move mit left button at: ( %d : %d )" % (event.x(), event.y()))
            # Todo rotate
            self.rotateEvent.emit(dxy)

        elif event.buttons() == Qt.RightButton:
            print(
                "Mouse move with right button at: ( %d : %d )" % (event.x(), event.y())
            )
            # Todo pan
            self.panEvent.emit(dxy)

        elif event.buttons() == Qt.MiddleButton:
            print(
                "Mouse move with middle button at: ( %d : %d )" % (event.x(), event.y())
            )
            # Todo zoom
            self.zoomEvent.emit(dxy)

        # update last position
        self.lastpos = currpos

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # print("Mouse pressed at: ( %d : %d )" % (event.x(), event.y()))
        self.lastpos = event.position()
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        # print("Mouse released at: ( %d : %d )" % (event.x(), event.y()))
        return super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        print("Mouse wheel at: ( %d : %d )" % (event.x(), event.y()))
        return super().wheelEvent(event)


class QImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0

        self.imageLabel = MouseTracker()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        # self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)
        self.imageLabel.setVisible(True)
        # self.imageLabel.underMouse.connect(self.dblClick)

        # self.scrollArea = QScrollArea()
        # self.scrollArea.setBackgroundRole(QPalette.Dark)
        # self.scrollArea.setWidget(self.imageLabel)
        # self.scrollArea.setVisible(False)

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.West)
        tabs.setMovable(True)

        for n, color in enumerate(["red", "green", "blue", "yellow"]):
            tabs.addTab(QLabel(color), color)

        layout = QHBoxLayout()
        layout.addWidget(self.imageLabel)
        layout.addWidget(tabs)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Image Viewer")
        self.resize(800, 600)

    def dblClick():
        print("double click")

    def open_QImage(self):
        options = QFileDialog.Options()
        # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "Images (*.png *.jpeg *.jpg *.bmp *.gif)",
            options=options,
        )
        if fileName:
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(
                    self, "Image Viewer", "Cannot load %s." % fileName
                )
                return

            self.imageLabel.setPixmap(QPixmap.fromImage(image))
            self.scaleFactor = 1.0

            # self.scrollArea.setVisible(True)
            self.printAct.setEnabled(True)
            self.fitToWindowAct.setEnabled(True)
            self.updateActions()

            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()

    def open_cv2(self):
        # headless:
        renderer = Renderer((1024, 1024))
        camera = Camera(1.0)

        # perspectives of the light field
        shots = [
            Shot(
                r"data\debug_scene\0000.png",
                [0, 0, 0],
                [0, 0, 0, 1],
                shot_fovy_degrees=60.0,
            ),
            Shot(
                r"data\debug_scene\0001.png",
                [0.2, 0, 0],
                [0, 0, 0, 1],
                shot_fovy_degrees=60.0,
            ),
            Shot(
                r"data\debug_scene\0014.png",
                [1.0, -1.0, 1.0],
                [0.13052618503570557, 0.0, 0.0, 0.9914448857307434],
                shot_fovy_degrees=60.0,
            ),
        ]

        shots = alfr.load_shots_from_json(
            r"data\debug_scene\blender_poses.json", fovy=60.0
        )

        for i, shot in enumerate(shots):

            vcam = {
                "mat_projection": (camera.mat_projection),
                "mat_lookat": (camera.mat_lookat),
            }

            img = renderer.project_shot(shot, vcam)

        # convert to uint8 and only use 3 channels (RGB)
        img = img[:, :, :3].astype("uint8")
        image = QImage(
            img.data, img.shape[1], img.shape[0], QImage.Format_RGB888
        ).rgbSwapped()
        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        self.scaleFactor = 1.0

        # self.scrollArea.setVisible(True)
        self.printAct.setEnabled(True)
        self.fitToWindowAct.setEnabled(True)
        self.updateActions()

        if not self.fitToWindowAct.isChecked():
            self.imageLabel.adjustSize()

    def open_cv2_old(self):
        options = QFileDialog.Options()
        # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "Images (*.png *.jpeg *.jpg *.bmp *.gif)",
            options=options,
        )
        if fileName:
            img = cv2.imread(fileName)
            if img is None:
                QMessageBox.information(
                    self, "Image Viewer", "Cannot load %s." % fileName
                )
                return

            img = img.astype("uint8")
            image = QImage(
                img.data, img.shape[1], img.shape[0], QImage.Format_RGB888
            ).rgbSwapped()
            self.imageLabel.setPixmap(QPixmap.fromImage(image))
            self.scaleFactor = 1.0

            # self.scrollArea.setVisible(True)
            self.printAct.setEnabled(True)
            self.fitToWindowAct.setEnabled(True)
            self.updateActions()

            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        # self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QMessageBox.about(
            self,
            "About Image Viewer",
            "<p>The <b>Image Viewer</b> example shows how to combine "
            "QLabel and QScrollArea to display an image. QLabel is "
            "typically used for displaying text, but it can also display "
            "an image. QScrollArea provides a scrolling view around "
            "another widget. If the child widget exceeds the size of the "
            "frame, QScrollArea automatically provides scroll bars.</p>"
            "<p>The example demonstrates how QLabel's ability to scale "
            "its contents (QLabel.scaledContents), and QScrollArea's "
            "ability to automatically resize its contents "
            "(QScrollArea.widgetResizable), can be used to implement "
            "zooming and scaling features.</p>"
            "<p>In addition the example shows how to use QPainter to "
            "print an image.</p>",
        )

    def createActions(self):
        self.openAct = QAction(
            "&Open...", self, shortcut="Ctrl+O", triggered=self.open_cv2
        )
        self.printAct = QAction(
            "&Print...", self, shortcut="Ctrl+P", enabled=False, triggered=self.print_
        )
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoomInAct = QAction(
            "Zoom &In (25%)",
            self,
            shortcut="Ctrl++",
            enabled=False,
            triggered=self.zoomIn,
        )
        self.zoomOutAct = QAction(
            "Zoom &Out (25%)",
            self,
            shortcut="Ctrl+-",
            enabled=False,
            triggered=self.zoomOut,
        )
        self.normalSizeAct = QAction(
            "&Normal Size",
            self,
            shortcut="Ctrl+S",
            enabled=False,
            triggered=self.normalSize,
        )
        self.fitToWindowAct = QAction(
            "&Fit to Window",
            self,
            enabled=False,
            checkable=True,
            shortcut="Ctrl+F",
            triggered=self.fitToWindow,
        )
        self.aboutAct = QAction("&About", self, triggered=self.about)
        self.aboutQtAct = QAction("About &Qt", self, triggered=qApp.aboutQt)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(
            int(factor * scrollBar.value() + ((factor - 1) * scrollBar.pageStep() / 2))
        )


if __name__ == "__main__":
    import sys
    import os
    from PySide6.QtWidgets import QApplication

    # os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication(sys.argv)
    # app.setAttribute(Qt.AA_EnableHighDpiScaling)
    imageViewer = QImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())
    # TODO QScrollArea support mouse
    # base on https://github.com/baoboa/pyqt5/blob/master/examples/widgets/imageviewer.py
    #
    # if you need Two Image Synchronous Scrolling in the window by PyQt5 and Python 3
    # please visit https://gist.github.com/acbetter/e7d0c600fdc0865f4b0ee05a17b858f2
