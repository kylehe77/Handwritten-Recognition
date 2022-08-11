from email.charset import QP
import os
import sys 
from PyQt5.QtWidgets import QComboBox,QSlider,QLineEdit,QDialog,QTextEdit,QLabel,QMessageBox,QMainWindow, QApplication, QPushButton, QWidget, QAction, QMenuBar, QMenu, qApp, QLabel, QMessageBox, QProgressBar, QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene
from PyQt5.QtGui import QIcon, QPainter, QPen, QPixmap, QTextCursor,QFont
from PyQt5.QtCore import pyqtSlot, QCoreApplication, Qt, QObject, pyqtSignal, QThread
from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox,QPainter, QPoint, QPen
from PyQt5 import QtWidgets
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from graphicsViewUI import graphicsView
from dataset import *
import cv2
import re
import json
from glob import glob
from quick_image import search
from look_by_class import *

'''
Reference code:
https://blog.csdn.net/u011389706/article/details/81460820
https://blog.csdn.net/weixin_43594279/article/details/116381021
https://www.cnblogs.com/PyLearn/p/7689170.html
https://pythonspot.com/pyqt5-tabs/
https://www.youtube.com/watch?v=4B3kYF5BhB4
https://www.geeksforgeeks.org/how-to-use-glob-function-to-find-files-recursively-in-python/
'''

clear = lambda: os.system('cls')
clear()

Current_use_model_name = "models/model.pth"

class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.train_byclass_data = ""
        self.test_byclass_data = ""
        self.initUI()

    #Initializing the main UI
    def initUI(self):
        importAction = QAction('Import Dataset', self)
        trainAction = QAction('Train Model', self)
        exitAction = QAction('Exit', self)
        exitAction.triggered.connect(qApp.quit)
        importAction.triggered.connect(self.importDialog)
        trainAction.triggered.connect(self.import_trainWidget)

        #Adding menubar at the top
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(importAction)
        fileMenu.addAction(trainAction)
        fileMenu.addAction(exitAction)

        #Adding the sub menus at the top
        predictConfig = QAction('Predict config', self)
        predictView = QAction('Predict view', self)
        viewMenu = menubar.addMenu('View')
        
        #Adding action to the menus
        viewMenu.addAction(predictConfig)
        viewMenu.addAction(predictView)
        predictConfig.triggered.connect(self.openPredictConfig)
        predictView.triggered.connect(self.openPredictView)

        #Setting window size and title
        self.resize(500, 500)
        self.move(500,300)
        self.setWindowTitle('Handwritten Text Recognizer')

        #Adding the center widget to the main window
        self.center_widget = centerWidegt(self)
        self.setCentralWidget(self.center_widget)
        self.show()
    
    #When train button pressed, activate train widget
    def import_trainWidget(self):
        trainconfigUI = trainWidget(self)
        trainconfigUI.show()

    #When import button pressed, activate import widget
    def importDialog(self):
        downloadUI = downloadEmnist(self)
        downloadUI.show()
        
    #When View button pressed, activate View widget
    def openPredictView(self):
        predictView = predictViewUI(self)
        predictView.show()
        
    #When Config button pressed, activate config widget
    def openPredictConfig(self):
        predictConfig = predictConfigUI(self)
        predictConfig.show()

#Class to set the main window widget
class centerWidegt(QWidget):

    def __init__(self, Parent=None):
        super().__init__(Parent)
        #Add buttons
        self.btn_clear = QPushButton('Clear')
        self.btn_recognize = QPushButton('Recognize')
        self.myCanvas = MyCanvasWidget(self)
        layout1 = QtWidgets.QHBoxLayout()
        layout1.addWidget(self.myCanvas)
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.btn_recognize)
        layout2.addWidget(self.btn_clear)
        layout3 = QtWidgets.QVBoxLayout()
        layout3.addLayout(layout1)
        layout3.addLayout(layout2)
        
        #Setting layout for the main window
        self.setLayout(layout3)
        self.btn_clear.clicked.connect(self.myCanvas.clear)
        self.btn_recognize.clicked.connect(self.myCanvas.recognize_byclass)
        
        
#Class to extract terminal messages to the GUI
class EmittingStream(QObject):

    textWritten = pyqtSignal(str)

    def write(self,text):
        self.textWritten.emit(str(text))
    
    def flush(self):
        pass
 

class downWorkThread(QThread):

    #New thread for downloading to stop the GUI from lagging/crashing
    def __init__(self, parent = None):
        super(downWorkThread,self).__init__(parent)
        self._parent = parent


    def run(self):
        down_datasets()
        cursor = self._parent.progressTextedit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText('\n')
        

class downloadEmnist(QDialog):

    #Layout of download/import UI
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Download Dataset') #Setting Window Name
        self.progressLabel = QLabel('Download Progress:') #Labelling progress
        self.progressTextedit = QTextEdit() #

        #progress bar widget
        self.progressLabelBar = QLabel('Progress')
        self.progressBar = QProgressBar(self)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.progressBar.setValue(0)
        self.downloadButton = QPushButton('Download')
        self.exitButton = QPushButton('Stop')

        #Configuring layout for the import UI
        layout1 = QtWidgets.QVBoxLayout()
        layout1.addWidget(self.progressLabel)
        layout1.addWidget(self.progressTextedit)
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.progressLabelBar)
        layout2.addWidget(self.progressBar)
        layout2.addWidget(self.downloadButton)
        layout2.addWidget(self.exitButton)
        layout3 = QtWidgets.QVBoxLayout()
        layout3.addLayout(layout1)
        layout3.addLayout(layout2)
        self.setLayout(layout3)

        #Connecting button to download function
        self.downloadData = downWorkThread(self)   
        self.downloadButton.clicked.connect(self.downloadStart)
        self.exitButton.clicked.connect(lambda :self.downloadData.terminate())
        
        #Displaying info onto the UI
        sys.stdout = EmittingStream(textWritten = self.normalOutputWritten)
        sys.stderr = EmittingStream(textWritten = self.normalOutputWritten)

        self.resize(400, 300)

    #start downloading/importing
    def downloadStart(self):
        self.downloadData.start()


    #Adding information to text edit
    def normalOutputWritten(self, text):
        cursor = self.progressTextedit.textCursor()
        cursor.movePosition(QTextCursor.End)

        cmd_data = str(text)
                
        #Writing to progress bar and the text editor to show information about downloading
        try:
            text = int(eval(cmd_data[:-1]))
            cursor.insertText(cmd_data)
            self.progressBar.setValue(text)
        except Exception as e:
            cursor.insertText(cmd_data)
        self.progressTextedit.setTextCursor(cursor)
        self.progressTextedit.ensureCursorVisible()


class trainWorkThread(QThread):

    #New thread for the training part to stop the GUI from lagging/crashing
    def __init__(self,parent=None):
        super(trainWorkThread,self).__init__(parent)
        self._parent = parent

    def run(self):  
        train_dnn_model(self._parent.train_loader,self._parent.test_loader,self._parent.dict_train)
        cursor = self._parent.progress_textedit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText("\n")

class trainWindow(QDialog):
#Class for the train window
    def __init__(self, parent = None):
        super().__init__(parent)
        self._parent = parent
        self.setWindowTitle('Train')
        self.train_loader,self.test_loader  = "",""
        self.dict_train = {}
        self.progress_label = QLabel('Progress:')
        self.progress_textedit = QTextEdit()
        self.progress_bar_label = QLabel('Progress')
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.download_button = QPushButton('Download')
        self.train_button = QPushButton('Train')
        self.exit_button = QPushButton('Cancel')

        layout1 = QtWidgets.QVBoxLayout()
        layout1.addWidget(self.progress_label)
        layout1.addWidget(self.progress_textedit)
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.progress_bar_label)
        layout2.addWidget(self.progress_bar)
        layout2.addWidget(self.download_button)
        layout2.addWidget(self.train_button)
        layout2.addWidget(self.exit_button)
        layout3 = QtWidgets.QVBoxLayout()
        layout3.addLayout(layout1)
        layout3.addLayout(layout2)
        self.setLayout(layout3)
        self.get_config_dict()

        #Connecting the buttons to the training function
        self.trainWork = trainWorkThread(self)    
        self.download_button.clicked.connect(self.load_datasets)
        self.train_button.clicked.connect(self.start_train_model)
        self.exit_button.clicked.connect(lambda :self.trainWork.terminate())
        self.resize(500,300)

        #Displaying info onto the UI
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        sys.stderr = EmittingStream(textWritten=self.normalOutputWritten)


    #Function to write the training information onto the GUI
    def normalOutputWritten(self, text):
        cursor = self.progress_textedit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cmd_data = str(text)
        
        #Writing the progress bar value and the information on the text editor visible on the GUI
        try:
            out_text = re.findall("Train Epoch:(\\d+).*?(\\d+.\\d+)%",cmd_data)[0]
            
            self.progress_bar.setValue(int((eval(out_text[1])/int(self._parent.get_config_dict()["EPOCHS"]))) + (int(eval(out_text[0]))-1)*100/int(self._parent.get_config_dict()["EPOCHS"])  )
            cursor.insertText(cmd_data)
        except Exception as e:
            cursor.insertText(cmd_data)

        self.progress_textedit.setTextCursor(cursor)
        self.progress_textedit.ensureCursorVisible()

    #Download dataset if not downloaded, if yes then print message
    def load_datasets(self):
        if (os.path.exists("data")):
            print ('Dataset already downloaded')
        else:
            app = downloadEmnist(self)
            app.show()
    
    #Getting the configuration from the dataset, and displaying it on the UI
    def get_config_dict(self):
        self.dict_train = self._parent.get_config_dict()
        return self.dict_train

    #Function to start the training process
    def start_train_model(self):
        self._parent.get_config_dict()
        if self.train_loader=="" or self.test_loader=="":
            self.train_loader,self.test_loader = down_datasets()
            self.train_loader,self.test_loader =  DataLoader_set(self.train_loader,self.test_loader,self.dict_train["BATCH_SIZE"])
            self.trainWork.start()
        else:
            QMessageBox.information(self,'Note', 'Please restart window')
        
class trainWidget(QDialog):

    def __init__(self, parent = None):
        super().__init__(parent)

        #Setting the widgets and the slider 
        self.setWindowTitle('Train config')
        self.train_config_label = QLabel('Train dataset')
        self.train_config_slider =  QSlider(Qt.Horizontal)
        self.train_config_slider_value = QLabel("100%")
        self.train_config_slider.setValue(100)
        self.train_config_slider.setMinimum(10)
        self.train_config_slider.setMaximum(100)
        self.train_config_slider.setSingleStep(1)
        self.train_config_slider.setTickPosition(QSlider.TicksBelow)
        
        #Batch size setup
        self.batch_size_label = QLabel('Batch size')
        self.batch_size_combox = QComboBox()
        self.batch_size_combox.addItems(['64','32','128','256','512'])

        #Epoch set up
        self.epochs_label = QLabel('Epochs')
        self.epochs_combox = QComboBox()
        self.epochs_combox.addItems(['5','10','15','20','30','50','100'])

        #Name set up
        self.train_config_model_name = QLabel('Model name')
        self.train_config_model_lineedit = QLineEdit('model.pth')
        self.train_button = QPushButton('Train')
        self.exit_button = QPushButton('Cancel')


        #Layout of the train widget
        layout1 = QtWidgets.QVBoxLayout()
        layout1.addWidget(self.train_config_label)
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.train_config_slider)
        layout2.addWidget(self.train_config_slider_value)
        layout3 = QtWidgets.QVBoxLayout()
        layout3.addWidget(self.train_config_model_name)
        layout3.addWidget(self.train_config_model_lineedit)
        layout4 = QtWidgets.QVBoxLayout()
        layout4.addWidget(self.batch_size_label)
        layout4.addWidget(self.batch_size_combox)
        layout5 = QtWidgets.QVBoxLayout()
        layout5.addWidget(self.epochs_label)
        layout5.addWidget(self.epochs_combox)

        layout6 = QtWidgets.QHBoxLayout()
        layout6.addWidget(self.train_button)
        layout6.addWidget(self.exit_button)
        layout7 = QtWidgets.QVBoxLayout()
        layout7.addLayout(layout1)
        layout7.addLayout(layout2)
        layout7.addLayout(layout3)
        layout7.addLayout(layout4)
        layout7.addLayout(layout5)
        layout7.addLayout(layout6)
        self.setLayout(layout7)

        #Connecting buttons onto the functions
        self.train_button.clicked.connect(self.start_train)
        self.exit_button.clicked.connect(self.close)
        self.train_config_slider.valueChanged.connect(self.valuechange)

        self.resize(300,200)
    
    #Getting config from the user's choosing 
    def get_config_dict(self):
        model_name = self.train_config_model_lineedit.text()
        batch_size = self.batch_size_combox.currentText()
        epochs = self.epochs_combox.currentText()
        dict_config = {
        "model_name":"models/"+model_name,
        "BATCH_SIZE":int(batch_size),
        "LEARN_RATE":1e-2,
        "EPOCHS":int(epochs)
        }
        return dict_config

    #Bringing up the training window when pressed
    def start_train(self):
        app = trainWindow(self)
        app.show()

    #Changing the train/test percentage
    def valuechange(self):
        size = self.train_config_slider.value()
        self.train_config_slider_value.setText(str(size)+"%")


class MyCanvasWidget(QWidget):

    def __init__(self, Parent = None):
        super().__init__(Parent)
        self.initData()  # Innitialize datas and the GUI
        self.initView()
        self.setWindowTitle("PEN")
    
    ## recognize the images from canvas
    def recognize_byclass(self):
        if(not os.path.exists("cache")): ## create a cache folder
            os.mkdir("cache")
            
        #Saving the image in the file to test later
        save_filename = "cache/test.png" #a cache image
        image = self.GetContentAsQImage()
        image.save(save_filename)

        #Reading the image, and predicting the result
        image = cv2.imread(save_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        new_image = 255 - image
        new_image = cv2.resize(new_image,(28,28))
        cv2.imwrite(save_filename,new_image) #replace the cache image
        dict_predict={
            "file_path":save_filename,
            "model_name":Current_use_model_name
         }
        if (not os.path.exists(Current_use_model_name)):
            QMessageBox.information(self,'Warning','No model selected')
            return
        #Predicting the image, and showing the result and probability
        probability,result = predict_image_singer_v1(dict_predict)
        dialog = QMessageBox(self)
        dialog.setWindowTitle('Result')
        dialog.setText('Result is: {} , Probability: {:.4f}'.format(result,probability))
        dialog.show()

   
    #Initializing the canvas
    def initData(self):
        self.size = QSize(480, 480)  #  Using QPIxmap to create a canvas and set the size
        self.board = QPixmap(self.size)
        self.board.fill(Qt.white)    #  Usinig white colour to fill out the backgroud of canvas

        self.isEmpty = True          # Set empty canvas as default      

        self.lastPos = QPoint(0, 0)  #  the previous position of mouse
        self.currentPos = QPoint(0, 0)  # current postition
        
        self.painter = QPainter()  # Use Qpainter
        self.thickness = 20  # set the default thickness
        self.penColor = QColor('black') #set the default colour
    

    def initView(self):
        # The size of GUI
        self.setFixedSize(self.size)

    def clear(self):
        # clear canvas
        self.board.fill(Qt.white)
        self.update()
        self.isEmpty = True

    def isEmpty(self):
        # Returns whether the canvas is empty
        return self.IsEmpty

    def GetContentAsQImage(self):
        # get the content of canvas, but return as QImage
        image = self.board.toImage()
        return image

    def paintEvent(self, paintEvent):
        #  Paint event
        #  Use Qpainter for drawing
        #  Drawing should be done bewtween begin()and end()   
        
          
        self.painter.begin(self)
        self.painter.drawPixmap(0, 0, self.board)
        self.painter.end()

    def mousePressEvent(self, mouseEvent):
        # when the mouse is pressed, this current position should be saved as previous position
        self.currentPos = mouseEvent.pos()
        self.lastPos = self.currentPos

    def mouseMoveEvent(self, mouseEvent):
        # when the mouse is moving, it should be updated to current position, and connect it to previous position
        self.currentPos = mouseEvent.pos()
        self.painter.begin(self.board)
       
        self.painter.setPen(QPen(self.penColor, self.thickness))# set the colour and size of pen
        # draw lines
        self.painter.drawLine(self.lastPos, self.currentPos)
        self.painter.end()
        self.lastPos = self.currentPos
        self.update()  # update position

    def mouseReleaseEvent(self, mouseEvent):
        self.IsEmpty = False  # when the mouse is release, the canvas is not empty anymore

class predictConfigUI(QDialog):
    
    # Saving models and changing models when required/needed
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setWindowTitle('Predict Config')
        self.chose_model_label = QLabel('Model: ')
        self.chose_model_combox = QComboBox(self)
        files_pth = os.listdir('models')
        self.chose_model_combox.addItems(files_pth)

        #Buttons added in dialog for saving model or exiting
        self.btn_ok =  QPushButton('Commit')
        self.btn_cancel = QPushButton('Exit')

        #Setting layout for UI
        layout1 = QtWidgets.QHBoxLayout()
        layout1.addWidget(self.chose_model_label)
        layout1.addWidget(self.chose_model_combox)
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.btn_ok)
        layout2.addWidget(self.btn_cancel)
        layout3 = QtWidgets.QVBoxLayout()
        layout3.addLayout(layout1)
        layout3.addLayout(layout2)
        
        self.setLayout(layout3)
        self.resize(300,200)
        self.btn_ok.clicked.connect(self.submit_data)
        self.btn_cancel.clicked.connect(self.close)

    # submitting data to chosen model
    def submit_data(self):
        global Current_use_model_name
        QMessageBox.information(self,'Note','Have been submitted')
        Current_use_model_name = 'models/'+str(self.chose_model_combox.currentText())
        return str(self.chose_model_combox.currentText())

class predictViewUI(QDialog):

    def __init__(self, parent = None):
        super().__init__(parent)
        self.setWindowTitle('Predict View')

        #Adding widgets to the window
        self.predict_root_label = QLabel('Root: ')
        self.predict_root_lineedit = QLineEdit()
        self.predict_root_button = QPushButton('Load')

        self.image_label = graphicsView(self)

        self.digit_label = QLabel('Search')
        self.digit_lineedit = QLineEdit('')

        self.all_label = QLabel('Datasets')
        self.all_lineedit = QTextEdit()

        self.search_button = QPushButton('Go')

        self.btn_predict =  QPushButton('Predict')
        self.btn_cancel = QPushButton('Exit')

        self.chose_combox = QComboBox()
        self.chose_combox.addItems(['Train','Test'])

        # Search slider for images throughout the digits/letter dataset
        self.search_slider =  QSlider(Qt.Horizontal)
        self.search_slider_value = QLabel('1')
        self.search_slider.setValue(1)
        self.search_slider.setMinimum(1)
        self.search_slider.setMaximum(697932)
        self.search_slider.setSingleStep(1)

        # Layout of buttons/widgets
        layout1 = QtWidgets.QHBoxLayout()
        layout1.addWidget(self.predict_root_label)
        layout1.addWidget(self.predict_root_lineedit)
        layout1.addWidget(self.predict_root_button)
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.image_label)
        layout3 = QtWidgets.QHBoxLayout()
        layout3.addWidget(self.btn_predict)
        layout3.addWidget(self.btn_cancel)
        layout3.addWidget(self.search_slider)
        layout3.addWidget(self.search_slider_value)
        layout3.addWidget(self.chose_combox)

        layout4 = QtWidgets.QHBoxLayout()
        layout4.addWidget(self.digit_label)
        layout4.addWidget(self.digit_lineedit)

        layout5 = QtWidgets.QVBoxLayout()
        layout5.addWidget(self.all_label)
        layout5.addWidget(self.all_lineedit)

        layout6 = QtWidgets.QHBoxLayout()
        layout6.addWidget(self.search_button)



        layout_left = QtWidgets.QVBoxLayout()
        layout_left.addLayout(layout1)
        layout_left.addLayout(layout2)
        layout_left.addLayout(layout3)


        layout_right = QtWidgets.QVBoxLayout()
        layout_right.addLayout(layout5)
        layout_right.addLayout(layout4)
        layout_right.addLayout(layout6)

        layout_main = QtWidgets.QHBoxLayout()
        layout_main.addLayout(layout_left,4)
        layout_main.addLayout(layout_right,2)
        self.setLayout(layout_main)
        self.resize(600,400)

        #Connecting the widgets to the functions below
        self.predict_root_button.clicked.connect(self.chose_filename)
        self.search_slider.valueChanged.connect(self.valuechange)
        self.chose_combox.currentIndexChanged.connect(self.valuechange) 
        self.btn_predict.clicked.connect(self.predict_data)
        self.btn_cancel.clicked.connect(self.close)
        self.search_button.clicked.connect(self.search_image)
        self.show_info()
        
    #Function allows for the slider to work, to go through the different images in the dataset
    #Also are able to show images based on the training set or the testing set
    def valuechange(self):

        if self.chose_combox.currentText() == 'Train':
            self.search_slider.setMinimum(1)
            self.search_slider.setMaximum(697932)
            size = self.search_slider.value()
            self.search_slider_value.setText(str(size))
            search(int(size),'train') #Displaying training set
            fileName = 'cache/search.png'
            self.predict_root_lineedit.setText(fileName)
            self.image=QPixmap(fileName)           
            self.item = QGraphicsPixmapItem(self.image)
            self.item.setFlag(QGraphicsItem.ItemIsMovable) 
            self.image_label.graphicsView= QGraphicsScene()  
            self.image_label.graphicsView.addItem(self.item)
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setScene(self.image_label.graphicsView)
        else:
            self.search_slider.setMinimum(1)
            self.search_slider.setMaximum(116323)
            size = self.search_slider.value()
            self.search_slider_value.setText(str(size))
            search(int(size),'test') #Displaying testing set
            fileName = 'cache/search.png'
            self.predict_root_lineedit.setText(fileName)
            self.image=QPixmap(fileName)           
            self.item = QGraphicsPixmapItem(self.image)
            self.item.setFlag(QGraphicsItem.ItemIsMovable) 
            self.image_label.graphicsView= QGraphicsScene()  
            self.image_label.graphicsView.addItem(self.item)
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setScene(self.image_label.graphicsView)

    #Shows all the numbers of the dataset on the right side
    def show_info(self):
        with open('cache/train_datasets.txt','r') as f:
            train_data = json.load(f)
        with open('cache/test_datasets.txt','r') as f:
            test_data = json.load(f)
        self.all_lineedit.clear()

        str_text = ''
        str_text = str_text+ 'Train Dataset\n'
        for key,value in train_data.items():
            str_text = str_text + (str(key)+':'+str(value)+'\n')
        str_text = str_text + 'Test Dataset\n'
        for key,value in test_data.items():
            str_text = str_text + (str(key)+':'+str(value)+'\n')
        self.all_lineedit.setText(str_text)

    #Searching the specified image in the line editor, and bringing it up on the canvas
    def search_image(self):
        try:
            search_filter = self.digit_lineedit.text()
            number = get_number(search_filter)
            index = filter_datasets(number,'test')
            train_data,test_data = down_datasets()
            look_image_content(test_data,index)
            fileName = 'cache/temp.png'
            self.predict_root_lineedit.setText(fileName)
            self.image=QPixmap(fileName)           
            self.item = QGraphicsPixmapItem(self.image)
            self.item.setFlag(QGraphicsItem.ItemIsMovable) 
            self.image_label.graphicsView= QGraphicsScene()  
            self.image_label.graphicsView.addItem(self.item)
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setScene(self.image_label.graphicsView)
        except Exception as e:
            QMessageBox.information(self,'Note','Error searching')
            
    #Predicting the images chosen from the files
    def predict_data(self):
        filename = self.predict_root_lineedit.text()
        dict_predict={
            'file_path':filename,
            'model_name':Current_use_model_name
         }

        #Choosing models to test
        if (not os.path.exists('models/model.pth')):
                QMessageBox.information(self,'Note', 'Please choose model')
                return

        #Predicting the image shown in the UI
        if os.path.isfile(filename):
            probability,result = predict_image_singer_v1(dict_predict)
            dialog = QMessageBox(self)
            dialog.setWindowTitle('Predict')
            dialog.setText('Result is: {}, probability: {:.4f}'.format(result, probability))
            dialog.show()
        else:
            all_files = glob(filename + '/*.png') + glob(filename + '/*.jpg')
            result_list = []
            for f in all_files:
                dict_predict={
                    'file_path':f,
                    'model_name':Current_use_model_name
                 }
                probability,result = predict_image_singer_v1(dict_predict)
                result_list.append([result,probability])

            #Bringing note if the image is not predictable
            if len(result_list) == 0:
                QMessageBox.information(self, 'Note', 'Not a predictable file')
                return


            dialog = QMessageBox(self)
            dialog.setWindowTitle('Predict')
            dialog.setText('Result is: '+str(result_list))
            dialog.show()

    #Choosing the file name to open the file with images
    def chose_filename(self):
        fileName,fileType = QtWidgets.QFileDialog.getOpenFileName(self, 'file', os.getcwd() + '/images', 
        'All Files(*);;Text Files(*.png)')
        self.predict_root_lineedit.setText(fileName)
        self.image=QPixmap(fileName)           
        self.item = QGraphicsPixmapItem(self.image)
        self.item.setFlag(QGraphicsItem.ItemIsMovable) 
        self.image_label.graphicsView= QGraphicsScene()  
        self.image_label.graphicsView.addItem(self.item)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScene(self.image_label.graphicsView)

class progressBar(QDialog):
    
    def __init__(self, parent = None):
        super().__init__(parent)
        self.initUI()
 
    def initUI(self):

        #Setting the progress bar values
        self.setWindowTitle('Progress')
        self.progressLabel = QLabel('Progress')
        self.progressBar = QProgressBar(self)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)

        #Adding the layouts of the progress bar
        layout1 = QtWidgets.QVBoxLayout()
        layout1.addWidget(self.progressLabel)
        layout1.addWidget(self.progressBar)

        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.progressLabel)
        layout2.addWidget(self.progressBar)
        self.setLayout(layout2)

        self.resize(300, 200)
        self.show()
    
    #Updating the value of the progress bar
    def update(self,value):
        self.progressBar.setValue(value)

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
