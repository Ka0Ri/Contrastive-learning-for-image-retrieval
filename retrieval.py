from utils.utils import get_device
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os
import numpy as np
import h5py
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import random
from models.BiTmodel import BiTSimCLR, BiTlinear
from models.ViTmodel import VisionTransformerSimCLR, VisionTransformerLinear
from models.Efficientmodel import EfficientCLR
from models.CGDmodel import CGDmodel

class ImageRetrieval(object):

    def __init__(self, dataset, config):

        self.config = config
        self.device = get_device()
        self.dataset = dataset
       
        self.train_loader, self.valid_loader = self.dataset.get_train_validation_data_loaders()
        self.test_loader = self.dataset.get_test_data_loaders()
        self.retrieval_config = config["Retrieval"]

       
        if(config['model_name'] == 'ViT'):
            model = VisionTransformerSimCLR(config).to(self.device)
        elif(config['model_name'] == 'BiT'):
            model = BiTSimCLR(config).to(self.device)
        elif(config['model_name'] == 'BiTlinear'):
            model = BiTlinear(config).to(self.device)
        elif(config['model_name'] == 'Eff'):
            model = EfficientCLR(config).to(self.device)
        elif(config['model_name'] == 'CGD'):
            model = CGDmodel(config).to(self.device)
        else:
            model = VisionTransformerLinear(config).to(self.device)
        

        self.model = self._load_pre_trained_weights(model)
        

    def _step(self, xis):
        # get the representations and the projections
        xis = xis.to(self.device)
        ris, zis = self.model(xis)  # [N,C]
        mask = torch.ones((ris.size(0), 100))
        if(self.retrieval_config["features"] == "projected"):
            f = zis
        elif(self.retrieval_config["features"] == "extracted"):
            f = ris
        else: #combined
            f = torch.cat([ris, zis], dim=1)
        return f, mask


    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.retrieval_config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found.")

        return model

    def extract_feature(self):

        
        with torch.no_grad():
            reps_vector = np.array([])
            cls = np.empty((0), np.int)
            self.model.eval()
          
            for [_, xis], cl in tqdm(self.train_loader):
                
                
                ris, mask = self._step(xis)
              
                if(reps_vector.size == 0):
                    reps_vector = np.empty((0, ris.size(-1)), np.float)
                   
                reps_vector = np.append(reps_vector, ris.cpu().numpy(), axis=0)
                cls = np.append(cls, cl.cpu().numpy(), axis=0)

            hf = h5py.File(os.path.join('./runs', self.retrieval_config['fine_tune_from'], self.retrieval_config['feas_name_file']), 'w')
            hf.create_dataset('reps', data=reps_vector)
            hf.create_dataset('targets', data=np.array(cls))
            
            hf.close()

            reps_vector = np.array([])
            mask_vector = np.array([])
            cls = np.empty((0), np.int)
            for [_, xis], cl in tqdm(self.test_loader):
                
                xis = xis.to(self.device)
                ris, mask = self._step(xis)
              
                if(reps_vector.size == 0):
                    reps_vector = np.empty((0, ris.size(-1)), np.float)
                    mask_vector = np.empty((0, mask.size(-1)), np.float)

                reps_vector = np.append(reps_vector, ris.cpu().numpy(), axis=0)
                mask_vector = np.append(mask_vector, mask.cpu().numpy(), axis=0)
                cls = np.append(cls, cl.cpu().numpy(), axis=0)

            hf = h5py.File(os.path.join('./runs', self.retrieval_config['fine_tune_from'], self.retrieval_config['feas_name_file'] + "_test"), 'w')
            hf.create_dataset('reps', data=reps_vector)
            hf.create_dataset('targets', data=np.array(cls))
            hf.create_dataset('mask', data=mask_vector)
            hf.close()
               
        
    def _scoring(self, features, cl, gallery, cls, topk=5, mode="L2"):

        if(mode == "L2"):
            d = torch.norm(gallery[None, :, :] - features[:, None, :], dim=-1)
            _, topk_index = torch.topk(d, topk, dim=-1, largest=False)
        else:
            features = F.normalize(features, dim=-1)
            gallery = F.normalize(gallery, dim=-1)
            d = torch.sum(gallery[None, :, :] * features[:, None, :], dim=-1)
            _, topk_index = torch.topk(d, topk, dim=-1, largest=True)

        predicted_cls = cls[topk_index]
        topk_accumulated = []
        average_precision = []
        for i in range(0, predicted_cls.size(0)):
            gt = cl[i].repeat(topk)
            matching = (gt == predicted_cls[i]).float()
            accumulated = torch.stack([matching[0:j + 1].sum() / (j + 1) for j in range(0, topk)])
            aver_pre = torch.sum(accumulated * matching) / (torch.sum(matching) + 10e-6)
            accuracy = [cl[i] in predicted_cls[i][:j+1] for j in range(0, topk)]

            topk_accumulated.append(accuracy)
            average_precision.append(aver_pre.cpu().numpy())
            
        return np.array(topk_accumulated), np.array(average_precision)

    def _retrieval(self, features, gallery, topk=5):
    
        d = np.linalg.norm(gallery - features, axis=-1)
        
        topk_index = d.argsort()[:topk]
        return topk_index

    def TopK_score_on_validation(self):

        topk = self.retrieval_config['topk']
        hf = h5py.File(os.path.join('./runs', self.retrieval_config['fine_tune_from'], self.retrieval_config['feas_name_file']), 'r')
        gallery = np.array(hf.get('reps'))
        gallery_cls = np.array(hf.get('targets'))

        gallery = torch.from_numpy(gallery).float().to(self.device)
        gallery_cls = torch.from_numpy(gallery_cls).long().to(self.device)


        topk_precision, mAP = np.zeros(topk), 0
        count  =  0
        with torch.no_grad():
            self.model.eval()

            for [xis, _], cl in tqdm(self.valid_loader):
                
                cl = cl.to(self.device).long()
               
                ris, _ = self._step(xis)
               
                topk_accumulated, average_precision = self._scoring(ris, cl, gallery, gallery_cls, 
                                                topk = topk, mode=self.retrieval_config['mode'])
                topk_precision += np.sum(topk_accumulated, axis=0)
               
                mAP += np.sum(average_precision)
                count += xis.size(0)
        
        print("Topk precision:", topk_precision / count)
        print("mean average precision", mAP / count)

    
    def draw_attention(self, img, mask):
        import math, cv2
        sz = int(math.sqrt(mask.shape[0]))
        mask = mask.reshape(sz, sz)
        img_sz = list(img.size())
        mask = cv2.resize(mask, (img_sz[1], img_sz[2]))
        mask = torch.tensor(mask)
        result = (mask * img)

        return result


    def random_retrieve(self, topk=5):
        
        hf = h5py.File(os.path.join('./runs', self.retrieval_config['fine_tune_from'], self.retrieval_config['feas_name_file'] + "_test"), 'r')
        gallery = np.array(hf.get('reps'))
        gallery_mask = np.array(hf.get('mask'))
        

        with torch.no_grad():
            self.model.eval()

            for [xis, _], cl in tqdm(self.valid_loader):
                
                query_index = random.randint(0, self.config["batch_size"] - 1)
                xis = xis.to(self.device)
                
                ris, mask = self._step(xis)
                feas = ris[query_index].cpu().numpy()
                retrieve_index = self._retrieval(feas, gallery, topk)
                retrieve_imgs = [self.draw_attention(xis[query_index].cpu(), mask[query_index].cpu().numpy())]
               
                for i in retrieve_index:
                    (retrieve_img, _), y = self.dataset.test_dataset[i]
                
                    masked_img = self.draw_attention(retrieve_img, gallery_mask[i])
                    
                    retrieve_imgs.append(masked_img)

                save_image(make_grid(retrieve_imgs, padding=2, normalize=True), 
                    os.path.join('./runs', self.retrieval_config['fine_tune_from'], 
                                    self.retrieval_config['retrieval_imagaes_name'] + "%s.png"%(cl[query_index])))

                
  




