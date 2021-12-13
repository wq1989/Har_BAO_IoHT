# encoding=utf-8
import os
import matplotlib.pyplot as plt
import models.network0 as net
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import torch.nn.functional as F


def eval_model(config, model, data_loader, loss_fn, device, nb_samples, mode = "val"):
	if mode == 'test':
	    config.LOGGER_FILE.info(f"{mode} (Size: {nb_samples})")
        
         
	start = time.time()
 
	model = model.eval()
	correct_predictions = 0
	losses = []
	imgs = []
	y_pred = []
	y_pred_probs = []
	y_test = []


	with torch.no_grad():
		for i, d in enumerate(data_loader):
			sample = d[0].to(device).float()
			targets = d[1].to(device).long()

			outputs = model(sample)
			_, preds = torch.max(outputs, dim=1)

			loss = loss_fn(outputs, targets)
			losses.append(loss.item())
			correct_predictions += torch.sum(preds == targets)
   
			imgs.extend(sample)
			y_pred.extend(preds)
			y_pred_probs.extend(F.softmax(outputs, dim=1)) # outputs
			y_test.extend(targets)   
   


	y_pred = torch.stack(y_pred).cpu()
	y_pred_probs = torch.stack(y_pred_probs).cpu()
	y_test = torch.stack(y_test).cpu()
 
 
	test_loss = np.mean(losses)
	test_acc = correct_predictions.double() / nb_samples
	if mode == 'test':
		config.LOGGER_FILE.info(f'Testing Loss: {test_loss}, Accuracy {test_acc.item()}, Time: {time.time()-start}')

 
	return test_acc, test_loss, imgs, y_pred, y_pred_probs, y_test


def train_epoch(model,data_loader,loss_fn,optimizer,device,n_examples, epoch, scheduler, iteration):
        
    
	model = model.train()
	losses = []
	y_pred = []
	y_test = []
	correct_predictions = 0
	num_steps_output = round(len(data_loader)/2) # we want 5 steps reports in each epoch

	for batch_idx, d in enumerate(data_loader):
		# iteration += 1
		sample = d[0].to(device).float()
		targets = d[1].to(device).long()

		#* predictions
		optimizer.zero_grad()
		output = model(sample)
  

		loss = loss_fn(output, targets) #torch.argmax(targets, 1)

		_, preds = torch.max(output, dim=1)

		correct_predictions_batch =  torch.sum(preds == targets)
		correct_predictions += correct_predictions_batch
		losses.append(loss.item())

		loss.backward()
  
		# Clip the norm of the gradients to 1.0.
		# This is to help prevent the "exploding gradients" problem.
		# Use it with RNN models
		# nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  
		# Update parameters and take a step using the computed gradient.
		# The optimizer dictates the "update rule"--how the parameters are
		# modified based on their gradients, the learning rate, etc.
		optimizer.step()
  
		# Update the learning rate.
		scheduler.step()
        
	return correct_predictions.double() / n_examples, np.mean(losses), iteration



def train_engine(config, model, optimizer, loss_fn, train_data_loader, val_data_loader, device):
    best_accuracy = 0
    best_loss = 999.0
    trials = 0
    # * Optimizer 
    total_steps = len(train_data_loader) * config.NR_EPOCHS
    scheduler = get_linear_schedule_with_warmup( optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    iteration = 0

    for epoch in range(config.NR_EPOCHS):
            start = time.time()


            #! Training	
            train_acc, train_loss, iteration = train_epoch(model,train_data_loader,loss_fn,optimizer,device,len(train_data_loader.dataset), epoch, scheduler, iteration)


            #! Validation	
            val_acc, val_loss, _,_,_,_ = eval_model(config, model,val_data_loader,loss_fn,device,len(val_data_loader.dataset), mode = "val")
            config.LOGGER_FILE.info(f'Ep [{epoch+1}/{config.NR_EPOCHS}], T_Loss: {train_loss:.4f}, T_Acc {train_acc:.4f}, Val_Loss: {val_loss:.4f}, Val_Acc {val_acc:.4f}, Time: {time.time()-start:.4f}')


    
            #! Best Model saving based on val_acc	
            if val_loss < best_loss:
                save_path = config.get_save_path(name=config.MODEL_NAME, 
                                                 dataset = config.DATA_NAME, 
                                                 seed=config.Curr_SEED, 
                                                 repeat=config.current_rep, 
                                                 extension = '.pt')
                torch.save(model, save_path)
                
                best_loss = val_loss
                trials = 0
            else:
                trials += 1
                if trials >= config.PATIENCE:
                    config.LOGGER_FILE.info(f">>> Early stopping on epoch {epoch}. <<<")
                    return model, val_acc, val_loss
            
    return model, val_acc, val_loss
