from utils.util import get_device
import torch
import torch.nn.functional as F
import os
import numpy as np
import h5py
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import random
from models.BiTmodel import BiTSimCLR
from models.ViTmodel import VisionTransformerSimCLR


class Resolution_retrieval(object):

    def __init__(self, dataset, config):

        self.config = config
        self.device = get_device()
        self.dataset = dataset
        self.test_loader = self.dataset.get_test_data_loaders()
        self.retrieval_config = config["Retrieval"]
        print("test dataset len: ", self.dataset.test_dataset.__len__())

       
        if(config['model_name'] == 'ViT'):
            model = VisionTransformerSimCLR(config).to(self.device)
        else:
            model = BiTSimCLR(config).to(self.device)

        self.model = self._load_pre_trained_weights(model)
        

    def _step(self, xis):
        # get the representations and the projections
        xis = xis.to(self.device)
        ris, zis, mask = self.model(xis)  # [N,C]
        # mask = torch.ones((ris.size(0), 100))
        f = ris
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
          
            reps_vector = np.array([])
            mask_vector = np.array([])
            cls = np.empty((0), np.int)
            idxs = np.empty((0), np.int)
            for [_, _, _, _, x], cl, idx in tqdm(self.test_loader):
                
                ris, mask = self._step(x)
              
                if(reps_vector.size == 0):
                    reps_vector = np.empty((0, ris.size(-1)), np.float)
                    mask_vector = np.empty((0, mask.size(-1)), np.float)

                reps_vector = np.append(reps_vector, ris.cpu().numpy(), axis=0)
                mask_vector = np.append(mask_vector, mask.cpu().numpy(), axis=0)
                cls = np.append(cls, cl.cpu().numpy(), axis=0)
                idxs = np.append(idxs, idx.cpu().numpy(), axis=0)

            hf = h5py.File(os.path.join('./runs', self.retrieval_config['fine_tune_from'], self.retrieval_config['feas_name_file'] + "_test"), 'w')
            hf.create_dataset('reps', data=reps_vector)
            hf.create_dataset('targets', data=np.array(cls))
            hf.create_dataset('indexes', data=np.array(idxs))
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


    def TopK_score(self):

        topk = self.retrieval_config['topk']
        hf = h5py.File(os.path.join('./runs', self.retrieval_config['fine_tune_from'], self.retrieval_config['feas_name_file'] + "_test"), 'r')
        gallery = np.array(hf.get('reps'))
        gallery_cls = np.array(hf.get('targets'))
        gallery_index = np.array(hf.get('indexes'))

        gallery = torch.from_numpy(gallery).float().to(self.device)
        gallery_cls = torch.from_numpy(gallery_cls).long().to(self.device)
        gallery_index = torch.from_numpy(gallery_index).long().to(self.device)

        with torch.no_grad():
            self.model.eval()
            topk_precision, mAP, count = np.zeros((4, topk)), np.zeros((4)), 0
           
            for xis, cl, idx in tqdm(self.test_loader):
                
                for s in range(4):

                    cl = cl.to(self.device).long()
                    idx = idx.to(self.device).long()
                    ris, _ = self._step(xis[s])
               
                    topk_accumulated, average_precision = self._scoring(ris, idx, gallery, gallery_index, 
                                                    topk = topk, mode=self.retrieval_config['mode'])
                    topk_precision[s] += np.sum(topk_accumulated, axis=0)
               
                    mAP[s] += np.sum(average_precision)
                count += cl.size(0)
        
        print("Topk precision:", topk_precision / count)
        print("mean average precision", mAP / count)           
    

    def _retrieval(self, features, gallery, topk=5):
    
        d = np.linalg.norm(gallery - features, axis=-1)
        topk_index = d.argsort()[:topk]
        return topk_index


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
        gallery_cls = np.array(hf.get('targets'))


        with torch.no_grad():
            self.model.eval()

            for xis, cl, idx in tqdm(self.test_loader):
                query_index = random.randint(0, self.retrieval_config["batch_size"] - 1)
                for s in range(4):
                    
                    ris, mask = self._step(xis[s])
                    feas = ris[query_index].cpu().numpy()
                    retrieve_index = self._retrieval(feas, gallery, topk)
                    retrieve_imgs = [self.draw_attention(xis[s][query_index].cpu(), mask[query_index].cpu().numpy())]
        
                    for i in retrieve_index:
                        (_, _, _, _, retrieve_img), y, id = self.dataset.test_dataset[i]
                        masked_img = self.draw_attention(retrieve_img, gallery_mask[i])
                        
                        retrieve_imgs.append(masked_img)

                    save_image(make_grid(retrieve_imgs, padding=2, normalize=True), 
                        os.path.join('./runs', self.retrieval_config['fine_tune_from'], 
                                        self.retrieval_config['retrieval_imagaes_name'] + "%s_%s.png"%(cl[query_index], s)))

                
  




