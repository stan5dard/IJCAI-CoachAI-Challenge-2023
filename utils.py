import os
import torch
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt


def draw_loss(record_train_loss, record_validation_loss, config):
    x_steps = range(1, config['epochs']+1, 20)
    fig = plt.figure(figsize=(12, 6))

    loss_train = record_train_loss['shot'][-1] + record_train_loss['area'][-1]
    loss_val = record_validation_loss['shot'][-1] + record_validation_loss['area'][-1]
    plt.title("{} loss".format(config['model_type']))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.ylim(0, 6)
    plt.xticks(x_steps)
    plt.grid()
    plt.plot(record_train_loss['total'], label='Train total loss')
    plt.plot(record_train_loss['shot'], label='Train shot CE loss')
    plt.plot(record_train_loss['area'], label='Train area NLL loss')
    plt.plot(record_validation_loss['total'], label='Validation total loss')
    plt.plot(record_validation_loss['shot'], label='Validation shot CE loss')
    plt.plot(record_validation_loss['area'], label='Validation area NLL loss')

    plt.legend()
    plt.savefig(f"{config['output_folder_name']}/train_loss_total_{loss_train}_val_loss_total_{loss_val}.png")
    plt.close(fig)

    x_steps = range(1, config['epochs']+1, 20)
    fig2 = plt.figure(figsize=(12, 6))
    plt.title("{} loss".format(config['model_type']))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.ylim(0, 6)
    plt.xticks(x_steps)
    plt.grid()
    # plt.plot(record_validation_loss['shot'], label='shot')
    # plt.plot(record_validation_loss['area'], label='area')
    plt.plot(record_validation_loss['height'], label='height')
    plt.plot(record_validation_loss['aroundhead'], label='aroundhead')
    plt.plot(record_validation_loss['backhand'], label='backhand')
    plt.plot(record_validation_loss['playerloc'], label='playerloc')
    plt.plot(record_validation_loss['opponentloc'], label='opponentloc')

    plt.legend()
    plt.savefig(f"{config['output_folder_name']}/validation.png")
    plt.close(fig2)
    
def draw_loss_cross(record_train_loss, record_validation_loss, config, fold):
    x_steps = range(1, config['epochs']+1, 20)
    fig = plt.figure(figsize=(12, 6))

    loss_train = record_train_loss['shot'][-1] + record_train_loss['area'][-1]
    loss_val = record_validation_loss['shot'][-1] + record_validation_loss['area'][-1]
    plt.title("{} loss".format(config['model_type']))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.ylim(0, 6)
    plt.xticks(x_steps)
    plt.grid()
    plt.plot(record_train_loss['total'], label='Train total loss')
    plt.plot(record_train_loss['shot'], label='Train shot CE loss')
    plt.plot(record_train_loss['area'], label='Train area NLL loss')
    plt.plot(record_validation_loss['total'], label='Validation total loss')
    plt.plot(record_validation_loss['shot'], label='Validation shot CE loss')
    plt.plot(record_validation_loss['area'], label='Validation area NLL loss')

    plt.legend()
    plt.savefig(f"{config['output_folder_name']}/train_loss_total_{loss_train}_val_loss_total_{loss_val}_fold_{fold}.png")
    plt.close(fig)


def save(encoder, decoder, config, loss, epoch=None):
    output_folder_name = f"{config['output_folder_name']}/"
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    if epoch is None:
        encoder_name = output_folder_name + 'encoder'
        decoder_name = output_folder_name + 'decoder'
        config_name = output_folder_name + 'config'
    else:
        # delete files in output_folder_name
        for file in os.listdir(output_folder_name):
            os.remove(output_folder_name + file)
        encoder_name = output_folder_name + 'encoder'
        decoder_name = output_folder_name + 'decoder'
        # encoder_name = output_folder_name + "epoch" + str(epoch) + "_loss" + str(round(loss, 3)) + 'encoder'
        # decoder_name = output_folder_name + "epoch" + str(epoch) + "_loss" + str(round(loss, 3)) + 'decoder'
        config_name = output_folder_name + 'config'
    
    torch.save(encoder.state_dict(), encoder_name)
    torch.save(decoder.state_dict(), decoder_name)
    with open(config_name, 'w') as config_file:
        config_file.write(str(config))
    with open(output_folder_name + "saved_epoch" + str(epoch), 'w') as epoch_marker:
        epoch_marker.write(str(epoch))

def save_fold(encoder, decoder, config, loss, fold, epoch=None):
    output_folder_name = f"{config['output_folder_name']}/"
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    if epoch is None:
        encoder_name = output_folder_name + 'encoder' + str(fold)
        decoder_name = output_folder_name + 'decoder' + str(fold)
        config_name = output_folder_name + 'config' + str(fold)
    else:
        # delete files in output_folder_name
        for file in os.listdir(output_folder_name):
            os.remove(output_folder_name + file)
        encoder_name = output_folder_name + 'encoder' + str(fold)
        decoder_name = output_folder_name + 'decoder' + str(fold)
        # encoder_name = output_folder_name + "epoch" + str(epoch) + "_loss" + str(round(loss, 3)) + 'encoder'
        # decoder_name = output_folder_name + "epoch" + str(epoch) + "_loss" + str(round(loss, 3)) + 'decoder'
        config_name = output_folder_name + 'config' + str(fold)
    
    torch.save(encoder.state_dict(), encoder_name)
    torch.save(decoder.state_dict(), decoder_name)
    with open(config_name, 'w') as config_file:
        config_file.write(str(config))
    with open(output_folder_name + "saved_epoch" + str(epoch), 'w') as epoch_marker:
        epoch_marker.write(str(epoch))

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    # def save_checkpoint(self, val_loss, model):
    #     '''Saves model when validation loss decrease.'''
    #     if self.verbose:
    #         self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #     torch.save(model.state_dict(), self.path)
    #     self.val_loss_min = val_loss