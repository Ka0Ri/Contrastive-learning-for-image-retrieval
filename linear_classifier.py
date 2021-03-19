import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import torch
from models.BiTmodel import BiTlinear
from models.ViTmodel import VisionTransformerLinear
from models.CGDmodel import CGDmodel
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from thop import profile, clever_format

from utils.utils import get_device, count_parameters, save_config_file, AverageMeter, accuracy, set_bn_eval
import pytorch_warmup as warmup
from loss.CGDloss import LabelSmoothingCrossEntropyLoss, BatchHardTripletLoss

import sys
from tqdm import tqdm
import logging


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


class LinearClassifier(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.train_config = config["Classifier"]

        self.loss_config = config["GCD-loss"]
      
        
        self.class_criterion = LabelSmoothingCrossEntropyLoss(smoothing=self.loss_config['smoothing'],
                                                     temperature=self.loss_config['temperature']).to(self.device)
        self.feature_criterion = BatchHardTripletLoss(margin=self.loss_config['margin']).to(self.device)
        
        if(config['model_name'] == 'ViT'):
            model = VisionTransformerLinear(config).to(self.device)
        elif(config['model_name'] == 'BiT'):
            model = BiTlinear(config).to(self.device)
        elif(config['model_name'] == 'CGD'):
            model = CGDmodel(config).to(self.device)


        self.model = self._load_pre_trained_weights(model)
        num_params = count_parameters(self.model)
        logger.info("Total Parameter: \t%2.1fM" % num_params)
        

    def _step(self, xi, labels=None):
    
        xi = xi.to(self.device)
        labels = labels.to(self.device)
        features, probs = self.model(xi)
        loss = self.class_criterion(probs, labels) + self.feature_criterion(features, labels)
        acc = accuracy(probs, labels, topk=(1,))
        
        return acc, loss

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
        flops, params = profile(self.model, inputs=(torch.randn(1, 3, 224, 224).to(self.device),))
        flops, params = clever_format([flops, params])
        logger.info('# Model Params: {} FLOPs: {}'.format(params, flops))
        
        n_iter = 0
        valid_n_iter = 0
        best_valid_meter = 0
        meter = AverageMeter()

        for epoch_counter in range(self.train_config['epochs']):
            self.model.train()
            self.model.apply(set_bn_eval)

            epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (acc=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)

            for xis, labels in epoch_iterator:
                optimizer.zero_grad()

                acc, loss = self._step(xis, labels)
                meter.update(acc[0], self.config["batch_size"])
                

                if n_iter % self.train_config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_acc', acc[0], global_step=n_iter)

                if apex_support and self.train_config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Epochs) (acc=%2.5f)" % (epoch_counter, self.train_config['epochs'], meter.val)
                )

                # warmup for the first 10 epochs
                scheduler.step(scheduler.last_epoch+1)
                warmup_scheduler.dampen()

            # validate the model if requested
            if epoch_counter % self.train_config['eval_every_n_epochs'] == 0:
                valid_meter = self._validate(valid_loader)
                if valid_meter > best_valid_meter:
                    # save the model weights
                    best_valid_meter = valid_meter
                    torch.save(self.model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_meter', valid_meter, global_step=valid_n_iter)
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

        eval_meter = AverageMeter()

        logger.info("***** Running Validation *****")
        # validation steps
        with torch.no_grad():
            self.model.eval()
            epoch_iterator = tqdm(valid_loader,
                          desc="Validating... (acc=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)

            for xis, labels in epoch_iterator:
            
                acc, loss = self._step(xis, labels)
                eval_meter.update(acc[0], labels.size(0))
                epoch_iterator.set_description("Validating... (acc=%2.5f)" % eval_meter.val)
          
            logger.info("\n")
            logger.info("Validation Results")
            logger.info("Valid acc: %2.5f" % eval_meter.avg)
    
        return eval_meter.avg



