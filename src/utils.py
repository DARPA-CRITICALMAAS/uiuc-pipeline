import re
import os
import cv2
import sys
import logging
import numpy as np
from rich.progress import Progress

# ANSI Escape Codes
ANSI_CODES = {
    'red' : "\x1b[31;20m",
    'bold_red' : "\x1b[31;1m",
    'green' : "\x1b[32;20m",
    'yellow' : "\x1b[33;20m",
    'bold_yellow' : "\x1b[33;1m",
    'blue' : "\x1b[34;20m",
    'magenta' : "\x1b[35;20m",
    'cyan' : "\x1b[36;20m",
    'white' : "\x1b[37;20m",
    'grey' : "\x1b[38;20m",
    'reset' : "\x1b[0m",
}
class RichHandler(logging.Handler):
    def __init__(self, progress):
        super().__init__()
        self.progress = progress

    def emit(self, record):
        self.progress.console.print(record.getMessage())

class ColoredConsoleFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: ANSI_CODES['cyan'],
        logging.INFO: ANSI_CODES['green'],
        logging.WARNING: ANSI_CODES['yellow'],
        logging.ERROR: ANSI_CODES['red'],
        logging.CRITICAL: ANSI_CODES['bold_red']
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno)
        record.levelname = color + record.levelname + ANSI_CODES['reset']
        if record.levelno >= logging.WARNING:
            record.msg = color + record.msg + ANSI_CODES['reset']
        return logging.Formatter.format(self, record)

# Utility function for logging to file and sysout
def start_logger(logger_name, filepath, log_level=logging.INFO, console_log_level=None, use_color=True, writemode='a'):
    log = logging.getLogger(logger_name)

    # Create directory if necessary
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname) and dirname != '':
        os.makedirs(os.path.dirname(filepath))

    # Special handle for writing to 'latest' file
    if os.path.exists(filepath) and os.path.splitext(os.path.basename(filepath.lower()))[0] == 'latest':
        if not os.stat(filepath).st_size == 0: # Empty file
            # Rename previous latest log
            with open(filepath) as fh:
                newfilename = '{}_{}.log'.format(*(fh.readline().split(' ')[0:2]))
                newfilename = newfilename.replace('/','-').replace(':','-')            
            os.rename(filepath, os.path.join(dirname, newfilename))

    # Formatter
    lineformat = '%(asctime)s %(levelname)s - %(message)s'
    #lineformat = '%(asctime)s %(levelname)s %(filename)s:(%(lineno)d) - %(message)s'
    file_formatter = logging.Formatter(lineformat, datefmt='%d/%m/%Y %H:%M:%S')
    if use_color:
        stream_formatter = ColoredConsoleFormatter(lineformat, datefmt='%d/%m/%Y %H:%M:%S')
    else:
        stream_formatter = file_formatter

    # Setup File handler
    file_handler = logging.FileHandler(filepath, mode=writemode)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)
    log.addHandler(file_handler)

    # Setup Stream handler (i.e. console)
    if console_log_level is not None:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(stream_formatter)
        stream_handler.setLevel(console_log_level)
        log.addHandler(stream_handler)
        log.setLevel(min(log_level,console_log_level))
    else:
        log.setLevel(log_level)
    
    return log

def swap_console_handler(log, handler):
    orig_handler = log.handlers[1]
    handler.setLevel(orig_handler.level)
    handler.setFormatter(orig_handler.formatter)
    log.handlers[1] = handler
    return orig_handler

def boundingBox(array):
    array = np.array(array).astype(int)
    min_xy = [min(array, key=lambda x: (x[0]))[0], min(array, key=lambda x: (x[1]))[1]]
    max_xy = [max(array, key=lambda x: (x[0]))[0], max(array, key=lambda x: (x[1]))[1]]
    return [min_xy, max_xy]

def sanitize_filename(filename, repl='□'):
    return re.sub('[^0-9a-zA-Z ._-]+', repl, filename).strip().replace(' ', '_')

def mask(image, contours):
    """
    Mask image so only area inside contours is visable
    image : numpy array of shape (C,H,W)
    contours : list of numpy arrays of shape (N,2) where N is the number of points in the contour

    Returns:
    mask_image : numpy array of shape (C,H,W)
    """
    image = image.transpose(1,2,0)
    area_mask = np.zeros((image.shape[0], image.shape[1], 1), dtype='uint8')
    for area in contours:
        cv2.drawContours(area_mask, [area], -1, 255, -1)
    mask_image = cv2.bitwise_and(image, image, mask=area_mask)
    mask_image = mask_image.transpose(2,0,1)
    return mask_image

def crop(image, countours):
    """
    Crop image to the bounding box of the contours
    image : numpy array of shape (C,H,W)
    contours : list of numpy arrays of shape (N,2) where N is the number of points in the contour

    Returns:
    crop_image : numpy array of shape (C,H,W)
    offset : tuple of (x,y) of the cropped images offset from the orignal image's top left corner.
    """
    image = image.transpose(1,2,0)
    min_pt, max_pt = None, None
    for area in countours:
        area_min_pt, area_max_pt = boundingBox(area)
        if min_pt is None:
            min_pt = area_min_pt
            max_pt = area_max_pt
        else:
            min_pt = (min(min_pt[0], area_min_pt[0]), min(min_pt[1], area_min_pt[1]))
            max_pt = (max(max_pt[0], area_max_pt[0]), max(max_pt[1], area_max_pt[1]))

    crop_image = image[:, min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]]
    crop_image = crop_image.transpose(2,0,1)
    return crop_image, min_pt
