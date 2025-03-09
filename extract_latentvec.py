


import glob
import multiprocessing
import os
import cv2
import numpy as np
import torch
import torchvision
from src.models.VAE.models import VAEModel


class Extractor:
    def __init__(self, weight_path:str, device:str='cpu'):
        self.latent_dim = 1024
        self.cond_input_dim = 474
        self.cond_output_dim = 192
        self.device = device
        self.model = VAEModel(self.latent_dim)
        ckpt = torch.load(weight_path, map_location=self.device)
        print(self.model.load_state_dict(ckpt["weights"]))
        self.model.to(self.device)
        self.model.eval()


    def read_image(self, file_path:str, im_size:tuple=(224,224)):
        img=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0]!= im_size[0] or img.shape[1]!= im_size[1]:
            img = cv2.resize(img, im_size, interpolation = cv2.INTER_AREA)
        # HWC to CHW
        img_np=np.transpose(img, (2,0,1))
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_np)).to(self.device).float().div(255)
        return img_tensor
    
    
    def recong_image(self, img_path:str, outdir:str='outputs'):
        img_tensor = self.read_image(img_path)
        with torch.no_grad():
            mu, logvar = self.model.encoder(img_tensor.unsqueeze(0).to(self.device))
            z = self.model.reparameterize(mu, logvar)
            x_hat = self.model.decoder(z)
        os.makedirs(outdir, exist_ok=True)
        torchvision.utils.save_image(x_hat, os.path.join(outdir, os.path.basename(img_path)))

    def extract_feature(self, img_path:str):
        img_tensor = self.read_image(img_path)
        with torch.no_grad():
            mu, logvar = self.model.encoder(img_tensor.unsqueeze(0).to(self.device))
            latent_vec = torch.cat([mu, logvar], dim=1).squeeze(0).cpu().numpy()

        return latent_vec
    

def extract(pid:int, img_paths:list, weight_path:str, save_dir:str, device:str='cpu'):
    os.makedirs(save_dir, exist_ok=True)
    extractor = Extractor(weight_path, device)
    for idx, img_path in enumerate(img_paths):
        print(f"pid: {pid} | progress: {idx+1}/{len(img_paths)}")
        try:
            latent_vec = extractor.extract_feature(img_path)
            np.save(os.path.join(save_dir, os.path.basename(img_path)[:-4]+'.npy'), latent_vec)
        except:
            print(f"pid: {pid} | {img_path} failed")

def run(root_dir:str, weight_path:str, save_dir:str, device:str='cpu', num_processes:int=16):
    img_paths = glob.glob(os.path.join(root_dir, '*.png'))
    
    process = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=extract, args=(i, img_paths[i::num_processes], weight_path, save_dir, device))
        p.start()
        process.append(p)

    for p in process:
        p.join()
        
if __name__ == '__main__':
    # weight_path = r'logs\ae2\ckpt_epoch_36_step_127260.pth'
    # device = 'cuda'
    # extractor = Extractor(weight_path, device)


    # img_path = r"D:\Project\dataset_gen\dataset\51k\aligned\images\fec3bea0-e722-4dee-9272-aeb7da896682.png@2024-12-24-01-49-32-198.png"

    # vec = extractor.extract_feature(img_path)
    # print(vec)


    root_dir = r"C:\Users\14404\Project\FPM\data\aug_images"
    weight_path = r'exp\stage1_vae\weights\VAE_epoch_18_step_133974.pth'
    save_dir = r'C:\Users\14404\Project\FPM\data\aug_features'
    device = 'cuda'
    num_processes = 4
    run(root_dir, weight_path, save_dir, device, num_processes)

