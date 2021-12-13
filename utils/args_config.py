import os
import torch
from .logger import getLogger
from functools import lru_cache
import numpy as np
import random
from texttable import Texttable
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter, writer
import torch.optim as optim


class Config():
    def __init__(self, project_path, data_name, model_name = 'mobilenet_v2', batch_size = 40, nb_epochs = 10, reduce_train = None, repeats = 1, patience = 300):
        super().__init__()
        
        self.SEED = 1337
        self.Curr_SEED = 1337
        self.RUN_MODE = 'train' #eval : eval from a fine-tuned model saved on disk
        self.PROJECT_PATH = project_path
        self.DATA_NAME = data_name
        self.momemtum = 0.9
        self.repeats = repeats
        self.current_rep = 0
        self.SAVE_PATH = ''
        self.SAVE_PATH_NAME = '' 
        self.VAL_SPLIT = 0.10
        self.label_encoder = ""
        self.REDUCE_DATA = reduce_train # None
        self.net_params =     params = {
                                            "lstm_layers" : 2,
                                            "lstm_units" : 50,
                                            "hidden_size": 128, # 256
                                            "dropout": 0.2,
                                            "learning_rate": 0.005,
                                            'optimizer': optim.Adam
                                            }

        self.DATA_DIR = os.sep.join([self.PROJECT_PATH, 'data'])
        self.DATASET_DIR = os.sep.join([self.DATA_DIR, self.DATA_NAME])
        self.DATA_FILE = os.sep.join([self.DATASET_DIR, 'facebook_edge.csv'])

        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.TEMP_DIR = os.sep.join([self.PROJECT_PATH, 'temp'])
        self.MODEL_NAME = model_name
        self.DATA_LOADERS_DIR = os.sep.join([self.TEMP_DIR, self.DATA_NAME +'_'+self.MODEL_NAME+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')])

        self.tensorboard_runs = os.sep.join([self.DATA_LOADERS_DIR , "runs"])
        if not os.path.isdir(self.DATA_LOADERS_DIR):	
            os.makedirs(self.DATA_LOADERS_DIR)

        if not os.path.isdir(self.tensorboard_runs):	
            os.makedirs(self.tensorboard_runs)
                        
        self.writer = SummaryWriter(self.tensorboard_runs)
        self.LOGGER, self.LOGGER_FILE  = getLogger(self.DATA_LOADERS_DIR, mode = 'print_all')
        self.LABELS_NAMES = ['covid', 'noncovid']
        self.NUM_CLASSES = len(self.LABELS_NAMES)
        self.INPUT_SIZE = 0 # n_features
        self.BATCH_SIZE = batch_size
        self.NR_EPOCHS = nb_epochs
        self.PATIENCE = patience
        
        self.TRAIN_SIZE = 0
        self.VAL_SIZE = 0
        self.TEST_SIZE = 0
        self.TRAIN_LOADER = []
        self.VAL_LOADER = []
        self.TEST_LOADER = []

    
    @lru_cache(maxsize=4)
    def get_save_path(self, name="mobilenet_v2", dataset = 'face', seed=0, repeat=0, acc = 0.0, loss = 0.0, epoch = 0, extension = '.bin'):
        self.SAVE_PATH_NAME = f'{name}_{dataset}_Rep_{repeat}_Seed_{seed}{extension}'
        self.SAVE_PATH = os.sep.join([self.DATA_LOADERS_DIR, self.SAVE_PATH_NAME]) 
        
        return self.SAVE_PATH


    
    def write_configs(self):
        args = vars(self)
        keys = sorted(args.keys())
        confT = Texttable()
        confT.set_deco(Texttable.HEADER)
        confT.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
        confT.add_rows([["Parameter", "Value"]])
        # [k.replace("_", " ").capitalize(), args[k]] for k in keys]
        self.LOGGER_FILE.info("\nProject path: {} \nGPU: {} \nInitial configuration: \n{} ".format(self.PROJECT_PATH, self.DEVICE, confT.draw()))
            
    def seed_everything(self, seed):
        self.SEED = seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed_all(self.SEED)
        np.random.seed(self.SEED)
        random.seed(self.SEED)
        os.environ['PYTHONHASHSEED'] = str(self.SEED)