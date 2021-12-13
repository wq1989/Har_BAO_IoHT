# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import logging
import logging.config
import os, sys
from datetime import datetime

def getLogger(data_loaders_folder, mode = 'root'):
    LOGFINENAME = '{0}_{1}.log'.format(os.path.basename(data_loaders_folder), datetime.now().strftime('%Y%m%d_%H%M%S'))
    LOGFILE = os.path.join(data_loaders_folder, LOGFINENAME)

    DEFAULT_LOGGING = {
        'version': 1,
        'formatters': {
            'deep_info': {
                'format': '%(asctime)s %(module)s(%(lineno)d) - %(levelname)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'standard': {
                'format': '%(asctime)s %(levelname)s(%(lineno)d): %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'simple': {
                'format': '%(message)s',
            },
        },
        
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'level': 'INFO',
                'stream': sys.stdout,
            },
            'file': {
                'class': 'logging.FileHandler',
                'formatter': 'deep_info', #simple
                'level': 'INFO',
                'filename': LOGFILE,
                'mode': 'w',  #overwrite
            },
        },
        
        'loggers': {
            'print_all': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False,
            },
            'root': {
                'level': 'INFO',
                'handlers': ['file'],
                'propagate': False,
            },
            __name__: {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False,
            },
        }
    }
    
    logging.config.dictConfig(DEFAULT_LOGGING)
    logger_console = logging.getLogger(mode)
    logger_file = logging.getLogger('root')

    return logger_console, logger_file