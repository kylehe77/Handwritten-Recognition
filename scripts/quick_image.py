import struct
import numpy as np
import cv2

'''
Reference code:
https://blog.csdn.net/XnCSD/article/details/87348750
'''


##After downloading ENMIST datasets, try to parse them o image files and display it  on GUI.   
def decode_idx3_ubyte(idx3_ubyte_file,index):
    with open(idx3_ubyte_file, 'rb') as f:
        print('Parsing documents:', idx3_ubyte_file)
        fb_data = f.read()

    offset = 0
    fmt_header = '>iiii'    # Read 4 unsinged int32 by big-endian method
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, fb_data, offset)
    
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(num_rows * num_cols) + 'B'

    # use struct module to read binary data and convert image pixels into numpy matrices
    for i in range(num_images):
        if(i+1)==index:
            im = struct.unpack_from(fmt_image, fb_data, offset)
            images = np.array(im).reshape((num_rows, num_cols))
            offset += struct.calcsize(fmt_image)
            break
        else:
            offset += struct.calcsize(fmt_image)
        

    return images

#this functioni used to search the image when move the silde bar in view interface
def search(index,mode="train"):
    if mode== "train":
        idx3_ubyte_file = "datasets/emnist-byclass-train-images-idx3-ubyte"
        images = decode_idx3_ubyte(idx3_ubyte_file,index).transpose(1,0)
        cv2.imwrite("cache/search.png",images) # theres a cache png , when the silde bar is moved, the png would be replaced. It can speed up retrieval.
    else:
        idx3_ubyte_file = "datasets/emnist-byclass-test-images-idx3-ubyte"
        images = decode_idx3_ubyte(idx3_ubyte_file,index).transpose(1,0)
        cv2.imwrite("cache/search.png",images)