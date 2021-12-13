# encoding=utf-8
import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from dataloader.load_dataset import load
import models.network0 as net
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from engine.training_engine import train_engine, eval_model
import random
from utils.args_config import Config
from utils.helper import summarize_results

def run_model(config):
    all_scores = list() 
    scores = list() 
    
    
    for r in range(config.repeats):
        config.current_rep = r + 1
        score = 0.0
        config.Curr_SEED = np.random.randint(1000)
        config.seed_everything(seed = config.Curr_SEED)
        
        if config.MODEL_NAME == "CNN":
            model = net.HARmodel(config).to(config.DEVICE) 
            
        if r == 0:
            config.LOGGER_FILE.info(model)
            
        optimizer = config.net_params['optimizer'](model.parameters(), lr=config.net_params["learning_rate"])
        loss_fn = nn.CrossEntropyLoss().to(config.DEVICE)
        
        mode = "train"
        if mode == "train":
            
            _, val_acc, val_loss = train_engine(config, model, optimizer, loss_fn, train_loader, val_loader, config.DEVICE)
            mode = "eval"
            
        if mode == "eval":
            #loading best model based on validation accuracy and evaluate using it
            loaded_model = torch.load(config.SAVE_PATH)
            loaded_model = loaded_model.to(config.DEVICE)
            score,_, imgs, y_pred, y_pred_probs, y_test = eval_model(config, loaded_model, test_loader, loss_fn, config.DEVICE, len(test_loader.dataset), mode = "test")
        
        score = score * 100.0
        config.LOGGER.info('> Rep_%d_Seed_%d: %.3f' % (r + 1, config.Curr_SEED, score))
        scores.append(score.item())
    all_scores.append(scores)
    config.LOGGER_FILE.info(f'Scores: {scores}')
    m, s = np.mean(scores), np.std(scores)
    config.LOGGER.info('Score: %.3f%% (+/-%.3f)' % (m, s))

    return score
    
    
  
if __name__ == '__main__':
      
    PROJECT_PATH = os.getcwd() 
    os.chdir(PROJECT_PATH)
    
    config = Config(PROJECT_PATH, 
                    data_name = 'wisdm', # UCI_HAR / wisdm
                    model_name = 'CNN', 
                    batch_size = 256, 
                    nb_epochs = 3000,
                    reduce_train = None,
                    patience = 50,
                    repeats = 5) 
    
    train_loader, test_loader, val_loader = load(config, batch_size=config.BATCH_SIZE)

    config.NUM_CLASSES = len(set(train_loader.dataset.labels))
    config.INPUT_SIZE = train_loader.dataset.samples.shape[2]
    config.write_configs()
    
    score  = run_model(config)
