import os
import sys
import logging

# Utility function for logging to file and sysout
def start_logger(logger_name, filepath, log_level, console=False, console_log_level=None, writemode='a'):
    if console_log_level is None:
        console_log_level = log_level
    
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
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    #log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s:(%(lineno)d) - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

    # Setup File handler
    file_handler = logging.FileHandler(filepath, mode=writemode)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(log_level)
    log.addHandler(file_handler)

    # Setup Stream handler (i.e. console)
    if console:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(log_formatter)
        stream_handler.setLevel(console_log_level)
        log.addHandler(stream_handler)

    log.setLevel(min(log_level,console_log_level))

    return log