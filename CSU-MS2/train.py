
import torch
import torch.distributed as dist
from model import ModelCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm
from transformers import AdamW
import logging
from torch.nn.parallel import DistributedDataParallel
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, os

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

torch.manual_seed(0)

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

class SimCLR(object):
    def __init__(self, dataset, config,device,world_size,rank):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])

    def _get_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Running on:", device)
        return device

    def train(self):
        #Dataloaders
        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = ModelCLR(**self.config["model_config"]).to(self.device)
        if self.config["smilesmodel_finetune"]:
            state_dict = torch.load(self.config["smilesmodel_finetune_path"], map_location=self.device)
            model.Smiles_model.load_state_dict(state_dict)
        else:
            model = self._load_pre_trained_weights(model)

        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.rank) 
            model = DistributedDataParallel(model, device_ids=[self.rank], find_unused_parameters = True)

        optimizer = torch.optim.AdamW(model.parameters(), 
                                        eval(self.config['learning_rate']), 
                                        weight_decay=self.config['weight_decay'])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                T_max=len(train_loader), 
                                                                eta_min=0, 
                                                                last_epoch=-1)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)


        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        #Checkpoint folder
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        if dist.get_rank() == 0:
            _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        print(f'Training...')

        for epoch_counter in range(self.config['epochs']):
            train_loader.sampler.set_epoch(epoch_counter)
            print(f'Epoch {epoch_counter}')
            train_loss = 0
            for (xis, mzs,intens,num_peaks)  in tqdm(train_loader):
                optimizer.zero_grad()
                xis = xis.to(self.device)
                mzs = mzs.to(self.device)
                intens = intens.to(self.device)
                num_peaks = num_peaks.to(self.device)
                
                # get the representations and the projections
                zis, zls = model(xis,mzs,intens,num_peaks)  # [N,C]

                loss = self.nt_xent_criterion(zis, zls)
                if dist.get_rank() == 0:
                    if n_iter % self.config['log_every_n_steps'] == 0:
                        self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                train_loss += loss.item()
                n_iter += 1
            if dist.get_rank() == 0:
                print('Training at Epoch ' + str(epoch_counter + 1) + 'with loss ' + str(train_loss))
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader, n_iter)
                if dist.get_rank() == 0:
                    print('Validation at Epoch ' + str(epoch_counter + 1) + 'with loss ' + str(valid_loss))
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    if dist.get_rank() == 0:
                        #torch.save(net.module, "saved_model.ckpt") 
                        torch.save(model.module.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                if dist.get_rank() == 0:
                    self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

        if self.world_size > 1:
            dist.destroy_process_group()

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs/', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader, n_iter):

        # validation steps
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            counter = 0
            print(f'Validation step')
            for (xis,mzs,intens,num_peaks)  in tqdm(valid_loader):

                xis = xis.to(self.device)
                mzs = mzs.to(self.device)
                intens = intens.to(self.device)
                num_peaks = num_peaks.to(self.device)

                # get the representations and the projections
                zis, zls = model(xis,mzs,intens,num_peaks)  # [N,C]

                loss = self.nt_xent_criterion(zis, zls)

                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
