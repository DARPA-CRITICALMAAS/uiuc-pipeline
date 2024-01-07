import os
import sys
import logging

class ColoredConsoleFormatter(logging.Formatter):
    # ANSI Escape Codes
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    bold_yellow = '\x1b[33;1m'
    blue = "\x1b[34;20m"
    grey = "\x1b[38;20m"
    reset = "\x1b[0m"
    
    LEVEL_COLORS = {
        logging.DEBUG: blue,
        logging.INFO: green,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno)
        record.levelname = color + record.levelname + self.reset
        if record.levelno >= logging.WARNING:
            record.msg = color + record.msg + self.reset
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
    file_formatter = logging.Formatter(lineformat, datefmt='%d/%m/%Y %H:%M:%S')
    if use_color:
        stream_formatter = ColoredConsoleFormatter(lineformat, datefmt='%d/%m/%Y %H:%M:%S')
    else:
        stream_formatter = file_formatter
    #log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s:(%(lineno)d) - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

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

    return log