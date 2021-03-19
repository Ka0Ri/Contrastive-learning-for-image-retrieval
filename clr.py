import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import torch
from models.BiTmodel import BiTSimCLR
from models.ViTmodel import VisionTransformerSimCLR
from models.Efficientmodel import EfficientCLR
from models.CGDmodel import CGDmodel
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.supconloss import SupConLoss
from utils.utils import get_device, count_parameters, save_config_file, AverageMeter, set_bn_eval
import pytorch_warmup as warmup

import sys
from tqdm import tqdm
import logging
import numpy as np


torch.manual_seed(0)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


class SimCLR(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.train_config = config["SimCLR"]

        self.loss_config = config['subcon-loss']
        self.criterion = SupConLoss(self.loss_config['temperature'],
                                    contrast_mode=self.loss_config['mode'],
                                    base_temperature=self.loss_config['base'],
                                    device=self.device).to(self.device)
        
        if(config['model_name'] == 'ViT'):
            model = VisionTransformerSimCLR(config).to(self.device)
        elif(config['model_name'] == 'Eff'):
            model = EfficientCLR(config).to(self.device)
        elif(config['model_name'] == 'CGD'):
            model = CGDmodel(config).to(self.device)
        else:
            model = BiTSimCLR(config).to(self.device)
        

        self.model = self._load_pre_trained_weights(model)
        num_params = count_parameters(self.model)
        logger.info("Total Parameter: \t%2.1fM" % num_params)
        

    def _step(self, xi, xj, labels=None):
    
        images = torch.cat([xi, xj], dim=0)
        images = images.to(self.device)
           
        bsz = self.config['batch_size']
        features, _ = self.model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        if self.loss_config["method"] == 'SupCon':
            labels = labels.to(self.device)
            loss = self.criterion(features, labels)
        elif self.loss_config["method"] == 'SimCRL':
            loss = self.criterion(features)
        
        return loss

    def train(self):

        #load data loader
        train_loader, valid_loader = self.dataset.get_train_validation_data_loaders()
   
        #define optimier
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), self.train_config['lr'], weight_decay=eval(self.train_config['weight_decay']))
        n_steps = self.train_config["epochs"] * len(train_loader)

        #learning rate schudler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

        if apex_support and self.config['fp16_precision']:
            self.model, optimizer = amp.initialize(self.model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        save_config_file(model_checkpoints_folder)
        logger.info("***** Running training *****")
        logger.info("  Total optimization steps = %d", n_steps)
        
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        losses = AverageMeter()

        for epoch_counter in range(self.train_config['epochs']):
            self.model.train()
            # self.model.apply(set_bn_eval)
            epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)

            for [xis, xjs], labels in epoch_iterator:
                optimizer.zero_grad()

                loss = self._step(xis, xjs, labels)
                losses.update(loss.item(), self.config["batch_size"])

                if n_iter % self.train_config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                if apex_support and self.train_config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Epochs) (loss=%2.5f)" % (epoch_counter, self.train_config['epochs'], losses.val)
                )

                # warmup for the first 10 epochs
                scheduler.step(scheduler.last_epoch+1)
                warmup_scheduler.dampen()

            # validate the model if requested
            if epoch_counter % self.train_config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

           
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.train_config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            logger.info("Loaded pre-trained model with success.")
        except FileNotFoundError:
            logger.info("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, valid_loader):

        eval_losses = AverageMeter()

        logger.info("***** Running Validation *****")
        # validation steps
        with torch.no_grad():
            self.model.eval()
            epoch_iterator = tqdm(valid_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)

            for [xis, xjs], labels in epoch_iterator:
            
                loss = self._step(xis, xjs, labels)
                eval_losses.update(loss.item(), self.config["batch_size"])
                epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
          
            logger.info("\n")
            logger.info("Validation Results")
            logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    
        return eval_losses.avg



