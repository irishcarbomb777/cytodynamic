import cv2 
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import copy
from random import randint
import uuid
import pandas as pd


class Image():
    """ This is an image class designed to give basic image functionality
    including showing, getting basic info, duplicate, crop, and perform 
    other basic functionality on images.
    """

    """Create class attributes"""
    cv_flags = {
        'gray'      :  0,
        'color'     :  1,
        'unchanged' : -1
    }

    def __init__( self, image_ndarray, image_name, color_model='gray' ):
        """ Initialize instance attributes to describe an image """
        # cv_flag             = Image.cv_flags.get(color_model, 'gray')
        """ Original Filename Implementation """
        # self.image_filename = image_filename
        # self.image          = cv2.imread(image_filename, cv_flag)
        self.image_name     = image_name
        self.image          = image_ndarray
        self.shape          = self.image.shape
        self.height         = self.shape[0]
        self.width          = self.shape[1]
        self.df             = pd.DataFrame(self.image)
        if len(self.shape) > 2:
            self.channel_cnt    = self.shape[2]
        else:
            self.channel_cnt    = 1
        if 1 < self.channel_cnt < 4:     # Set default colorspace to RGB
            self.image       = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.color_space = 'RGB'
        elif self.channel_cnt > 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2RGBA)
            self.color_space = 'RGBA'
        else:
            self.color_space = 'Gray'
        # self.histr = cv2.calcHist([self.image], [0], None, [256], [0,256])

    @classmethod
    def from_file(cls, image_filename, image_name, color_model='gray'):
        cv_flag             = Image.cv_flags[color_model]
        image               = cv2.imread(image_filename, cv_flag)
        return cls(image, image_name, color_model)
       
    
    def print_info(self):
        """ Print a statement with basic image info."""

        # Note: change Image Info to something more descriptive
        line_len = len(self.image_name) + 10
        print("-"*line_len)
        print("---- " + self.image_name + " ----")
        print("-"*line_len)
        # print("Original Filename : " + self.image_filename)
        print("Height            : " + str(self.height))
        print("Width             : " + str(self.width))
        print("Color Channels    : " + str(self.channel_cnt))
        print("Color Space       : " + self.color_space)
        print("")

    def cvt_color(self, color='RGB'):
        """ Create a convert color function keyword switch """

        def RGBtoBGR():
            self.image       = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            self.color_space = 'BGR'     
            self.shape       = self.image.shape 
            self.channel_cnt = self.shape[2]
        
        def RGBtoRGBA():
            self.image       = cv2.cvtColor(self.image, cv2.COLOR_RGB2RGBA)
            self.color_space = 'RGBA'
            self.shape       = self.image.shape
            self.channel_cnt = self.shape[2]

        def RGBtoGray():
            self.image       = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            self.color_space = 'Gray'
            self.shape       = self.image.shape
            self.channel_cnt = 1

        def BGRtoRGB():
            self.image       = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.color_space = 'RGB'
            self.shape       = self.image.shape        
            self.channel_cnt = self.shape[2]

        def RGBAtoRGB():
            self.image       = cv2.cvtColor(self.image, cv2.COLOR_RGBA2RGB)
            self.color_space = 'RGB'
            self.shape       = self.image.shape
            self.channel_cnt = self.shape[2]
        
        def RGBAtoGray():
            self.image       = cv2.cvtColor(self.image, cv2.COLOR_RGBA2GRAY)
            self.color_space = 'Gray'
            self.shape       = self.image.shape
            self.channel_cnt = 1
        
        def RGBAtoBGRA():
            self.image       = cv2.cvtColor(self.image, cv2.COLOR_RGBA2BGRA)
            self.color_space = 'BGRA'
            self.shape       = self.image.shape
            self.channel_cnt = self.shape[2]

        cvtColorSwitcher = {
            'rgb' : {'rgba': RGBtoRGBA,
                     'bgr' : RGBtoBGR,
                     'gray': RGBtoGray},
            'rgba': {'rgb' : RGBAtoRGB,
                     'gray': RGBAtoGray,
                     'bgra': RGBAtoBGRA},
            'bgr' : {'rgb' : BGRtoRGB}
        }

        function = cvtColorSwitcher[self.color_space.lower()][color.lower()]
        function()
        return self
        
    def im_show(self):
        """Show the image based on its current color mapping"""

        cmapSwitcher = {
            'gray' : 'gray',
            'rgb'  : 'viridis',
            'rgba' : 'viridis'
        }

        cmap = cmapSwitcher[self.color_space.lower()]
        plt.figure(self.image_name)
        plt.title('---- ' + self.image_name + ' ----')
        plt.imshow(self.image, cmap=cmap, vmin=0, vmax=255)

    def duplicate(self, image_name):
        """Duplicate the image and return an unlinked copy """
        self = copy.deepcopy(self)
        self.image_name = image_name
        return self

    def crop_image(self, start_x, start_y, end_x, end_y):
        """Crop the existing image and return the unlinked cropped image"""
        self = copy.deepcopy(self)
        self.image = self.image[start_y:end_y, start_x:end_x]
        self.image = self.image.astype(np.uint8)
        self.df    = pd.DataFrame(self.image)
        self.image_name = self.image_name + ' - Cropped'
        return self 
    
    def resize(self, new_w='', new_h='' ):
        """ Resize the image with new height/width to preserve aspect ratio """
        # Get Scale Values
        if isinstance(new_w, int):
            scale = new_w/self.width * 100
            self.image_name = self.image_name + ' - Resized W-'+str(new_w)
        if isinstance(new_h, int):
            scale = new_h/self.height * 100
            self.image_name = self.image_name + ' - Resized H-'+str(new_h)
        # Calculate new width & height
        width = int(self.image.shape[1] * scale / 100)
        height = int(self.image.shape[0] * scale / 100)
        dsize = (width, height)

        # Resize the image
        self.image = cv2.resize(self.image, dsize)

        # Set new class values
        return self 
    
    def bin_threshold(self, low_thresh, high_thresh):
        self = copy.deepcopy(self)
        ret, self.image = cv2.threshold(self.image, low_thresh, high_thresh,
                                            cv2.THRESH_BINARY)
        self.df = pd.DataFrame(self.image)                                    
        
        self.image_name = self.image_name + f' - Threshold({low_thresh}, {high_thresh})'
        return self
    
    def bin_erode(self, kernel, iterations):
        self            = copy.deepcopy(self)
        self.image      = cv2.erode(self.image, kernel, iterations=iterations)
        self.image_name = self.image_name + f' - Eroded'
        return self
    
    def bin_invert(self, new=True):
        if new:
            self            = copy.deepcopy(self)
            self.image      = cv2.bitwise_not(self.image)
            self.image_name = self.image_name + f' - Inverted'
            return self
        else:
            self.image = cv2.bitwise_not(self.image)

    def bin_opening(self,kernel):
        self = copy.deepcopy(self)
        self.image = cv2
         
    def to_csv(self, filename):
        self.df.to_csv(filename)

        
    def grab_area(self):
        # Get ROI of image
        (x_0, y_0, x_os, y_os) = cv2.selectROI("Original Image", self.g_xy,\
        True, False)

        # Crop image to ROI and create new image
        g_xy_crop = self.g_xy[ y_0:(y_0+y_os), x_0:(x_0+x_os) ]
        g_xy_crop = Waveform_2D(wave=g_xy_crop)
        return g_xy_crop       

    def show_hist(self):
        plt.figure('Image Histogram')
        plt.plot(self.histr)
        plt.xlim([0,256])

    def simple_stretch(self, L, H):
        stretch = copy.deepcopy(self)
        stretch.image = ((255/(H-L))*(stretch.image-L))
        stretch.image = stretch.image.astype(np.uint8)
        stretch.image[stretch.image > 255] = 255
        stretch.image[stretch.image < 0] = 0
        stretch.histr = cv2.calcHist([stretch.image], [0], None, [256], [0,256])
        return stretch
    
    
class SgGuiImage(Image):
    """ A class for image operations related to using PySimpleGui """

    def gui_resize_to_bytes(self, new_w='', new_h=''):
        """ Resize selected image and output a byte representation """
        temp = copy.deepcopy(self)
        temp = temp.resize(new_w, new_h)
        temp = temp.cvt_color(color='BGRA')
        is_success, im_buf_array = cv2.imencode(".png", temp.image)
        bytes_im = im_buf_array.tobytes()

        return bytes_im
