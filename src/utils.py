import os
import sys
import cv2
import logging

log = logging.getLogger('DARPA_CMASS_PIPELINE')

# Utility function for logging to file and sysout
def start_logger(filepath, debuglvl, writemode='a'):
    # Create directory if necessary
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname) and dirname != '':
        os.makedirs(os.path.dirname(filepath))

    # Special handle for writing to 'latest' file
    if os.path.exists(filepath) and os.path.splitext(os.path.basename(filepath.lower()))[0] == 'latest':
        # Rename previous latest log 
        with open(filepath) as fh:
            newfilename = '{}_{}.log'.format(*(fh.readline().split(' ')[0:2]))
            newfilename = newfilename.replace('/','-').replace(':','-')            
        os.rename(filepath, os.path.join(dirname, newfilename))

    # Formatter
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    #log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s:(%(lineno)d) - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

    # Setup File handler
    file_handler = logging.FileHandler(filepath, mode=writemode)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(debuglvl)

    # Setup Stream handler (i.e. console)
    #stream_handler = logging.StreamHandler(stream=sys.stdout)
    #stream_handler.setFormatter(log_formatter)
    #stream_handler.setLevel(logging.INFO)

    # Add Handlers to logger
    log.addHandler(file_handler)
    #log.addHandler(stream_handler)
    log.setLevel(debuglvl)

# Utility function for loading images
def safeLoadImg(filepath):
    if not os.path.exists(filepath):
        log.error('Image file "{}" does not exist'.format(filepath))
        return None
    img = cv2.imread(filepath)
    if img is None:
        log.error('Could not load {} as image.'.format(filepath))
        return None
    return img