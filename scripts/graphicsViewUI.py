from PyQt5.QtWidgets import QDialog,QApplication,QDesktopWidget,QGroupBox,QLabel,QLineEdit,QPushButton,QGridLayout,QGraphicsView,QGraphicsScene,QGraphicsItem,QGraphicsPixmapItem,QMessageBox
from glob import glob
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import sys
import random
import os
import json

'''
Reference Code:
https://www.youtube.com/watch?v=mcT_bK1px_g
https://www.youtube.com/watch?v=2ZGpaRyO-jE
https://www.youtube.com/watch?v=4B3kYF5BhB4
'''

# images added to file path, will rotate through/ be able to choose from all photos in the file
Config_dict = {"datasets_path":"images"}
Format_Support=(".jpg",".JPG",".PNG",".png",".JPEG",".jpeg")


class graphicsView(QGraphicsView):

    def __init__(self,parent=None):
        super().__init__(parent)

        #Setting up the basic functions for the UI
        self.setDragMode(2)
        self.setAcceptDrops(True)
        self._parent = parent
        self.zoomscale = 1
        self.pos = 0
        self.image_index = 0
        
    #Allows for double-clicking, and changing the photo inside the canvas
    def mouseDoubleClickEvent(self,event):

        data = Config_dict["datasets_path"]
        all_files = sorted(glob(data+"/*.png") + glob(data+"/*.jpg"))
        all_files_len = len(all_files)
        if self.image_index >= all_files_len:
            self.image_index = 0

        #Selects random image in the folder containing the images
        current_file = random.choice(all_files)
        self._parent.predict_root_lineedit.setText(current_file)

        format_corrent = 0
        for pic in Format_Support:
            if pic == os.path.splitext(current_file)[1]:
                format_corrent=1
        if format_corrent == 1 :
            self.image=QPixmap(current_file)
            self.graphicsView= QGraphicsScene()            
            self.item = QGraphicsPixmapItem(self.image)
            
            #Allows the image to be dragged around
            self.item.setFlag(QGraphicsItem.ItemIsMovable)
            self.graphicsView.addItem(self.item)
            self.setAlignment(Qt.AlignCenter)
            self.setScene(self.graphicsView)

        self.image_index += 1
    
    #These three events down below allow for drag, move and drop events to be happening inside the UI
    def dragEnterEvent(self, event):
            data = event.mimeData()
            urls = data.urls()
            if ( urls and urls[0].scheme() == 'file' or os.path.isfile(data.text()) ):
                event.acceptProposedAction()

    #The function should be the same as the dragEnterEvent
    def dragMoveEvent(self, event):
            data = event.mimeData()
            urls = data.urls()
            if ( urls and urls[0].scheme() == 'file' or os.path.isfile(data.text())):
                event.acceptProposedAction()

    #Unpacking the data that the event has received, and choosing the image from the folder to display on to the canvas
    def dropEvent(self, event):
            data = event.mimeData()
            urls = data.urls()

            #Opening the file path containing the images
            if ( urls and urls[0].scheme() == 'file' ):
                filepath = str(urls[0].path())[1:]
            elif(os.path.isfile(data.text())):
                filepath = data.text()
            else:
                return
            
            #Selecting the images from the folder, and allowing it to be displayed on the canvas and let it be moveable
            if (os.path.isdir(filepath)):
                format_corrent=0
                all_files = os.listdir(filepath)
                for file in all_files:
                    if file.endswith(Format_Support):
                        format_corrent = 1
                        break
                if (format_corrent == 1):
                    if (self._parent != None):
                        self._parent.predict_root_lineedit.setText(filepath)
                    self.graphicsView= QGraphicsScene()            
                    self.item = QGraphicsPixmapItem(self.image)
                    self.item.setFlag(QGraphicsItem.ItemIsMovable)
                    self.graphicsView.addItem(self.item)
                    self.setAlignment(Qt.AlignCenter)
                    self.setScene(self.graphicsView)
                else:
                    QMessageBox.information(self, 'Note', 'Menu does not include photos')


            else:
                format_corrent = 0
                if (self._parent != None):
                    self._parent.predict_root_lineedit.setText(filepath)
                
                for pic in Format_Support:
                    if pic == os.path.splitext(filepath)[1]:
                        format_corrent = 1
                if format_corrent == 1 :
                    self.image = QPixmap(filepath)
                    self.graphicsView = QGraphicsScene()            
                    self.item = QGraphicsPixmapItem(self.image)
                    
                    self.item.setFlag(QGraphicsItem.ItemIsMovable)
                    self.graphicsView.addItem(self.item)
                    self.setAlignment(Qt.AlignCenter)
                    self.setScene(self.graphicsView)
                else:
                    QMessageBox.information(self,'Note', 'Not a photo')
    
    #Allows for zooming in and out if the selected image
    def wheelEvent(self, event):
            angle = event.angleDelta()   
            angleX = angle.x()  
            angleY = angle.y()                  
  
            if angleY > 0:
                try:
                    self.zoomscale=self.zoomscale + 0.1
                    self.item.setScale(self.zoomscale)
                except:
                    pass
           
            elif angleY < 0:
                try:
                    self.zoomscale=self.zoomscale - 0.1
                    self.item.setScale(self.zoomscale)
                except:
                    pass
    
def main():
    app = QApplication(sys.argv)
    windows = graphicsView()
    windows.resize(700, 400)
    windows.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()