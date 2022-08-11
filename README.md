# Handwritten Digits/Letters Recognition
Written by Ray Xiang and Tianhua He （Team 16）<br />
This ia a project required us to design  and  implement  a  application  that recognizes  handwritten  digits  and  English  letters. This application was bulit as the Project 1 for COMPSYS 302 of University of Auckland.

## Windows System used! Windows 10 Pro, Version 21H2

## Installation Instructions
1. Install miniconda3  https://docs.conda.io/en/latest/miniconda.html (We use python 3.8 in our project)
2. Open miniconda3 and type the following: **conda create –n py38 python=3.8** (When Proceed([y]/n) shows up, type y and enter to install packages)
3. Once packages are installed, typed the following to active conda : **conda activate py38**. Your enviroment now should change from (base) to (py38)
4. Enter following command to install Pytorch :  **pip install PyQt5 torch torchvision**
5. Install other required libraries: PIL, numpy, OpenCV by enter the command: **pip install Pillow numpy opencv-python**
6. Download the  "script" folder
7. Download the extra folder "datasets" inside the script folder and unzip it: https://drive.google.com/drive/folders/1FM7YsFolqspbbeuDYEnnc9KTtEbi3E8T?usp=sharing
8. If the "datasets" folder is not inside the "scripts" folder, please move it inside the "script" folder. **Program only runs properly if the "datasets" folder is inside the "script" folder alongside "main.py" and the other files**

Package versions:
(PyQt5 - 5.16.6) <br />
(torch - 1.11.0) <br />
(torchvision - 0.12.0) <br />
(numpy - 1.22.3) <br />
(OpenCV - 4.5.5.64) <br />
(Pillow (PIL) - 9.1.0) <br />
(Python - 3.8) <br />


## How to run
* Make sure all packages alreday installed and the "datasets" folder has been installed inside the "script" folder. Also, **DO NOT** delete the files inside 'models' folder.<br />
* Run main.py<br />
* To open the download window, click 'File' and then 'Import Dataset' -> To download the dataset, click 'Download'<br />
* To open the train window, click 'File' and then 'Train Model'  
  * Use slide bar to split train dataset into train/validation<br />
  * Change model name, select batch size and epochs<br />
  * Click 'Train' button and the train window will pop up<br />
  * Click 'Download' to check if the dataset already download<br />
  * If the dataset is downloaded, the click 'Train' to start training<br />
  * If you want to stop training, click 'Cancel'.<br />
* To load a model for prediction, click "View" -> click "Predict Config" to choose which model you would like to use for prediction.<br />
* To recognize handwritten digits/letters, go back to main window and start writing on canvas<br />
  * click 'Recognize' to predict your handwritten digits/letters<br />
  * click 'Clear' to clear drawings on canvas<br />
* To open the image display and prediction window, click 'View' -> click 'Predict View'
   * You can see simple statistics of the dataset (the number of each character or digit)  at the right of window.
   * To load all the images you can either press 'enter' on your keyboard, or press the 'load' button at the top of the window. Then you will be able to scroll through <br />
   * You can also randomly skip images by double-clicking your mouse inside the canvas, then it will skip to a random photo in the file for prediction <br />
   * Once you have an image you want to predict the result, you can then press the 'predict' button at the bottom left for prediction <br />
   * You can also choose the train/test through the use of the combobox, which allows you to choose betweeen the training set and the testing set <br />
   * You can also slide through the slider, which then goes through all of the images throughout the dataset <br />
   * To the left here, there is a search button, where you can search individual characters when you type in the line editor, and press the 'Go' button <br />
