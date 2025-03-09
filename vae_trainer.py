import glob
import math
import os
from queue import Queue
import random
import re
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
from src.models.VAE.models import VAEModel
from src.dataset.datasets import ImageDataset
from src.models.VAE.losses import LPIPS
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed:int):
    seed = int(seed)
    seed = seed if seed != -1 else random.randrange(1 << 32)
    print(f"Set seed to {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except:
        pass
    return seed



class Trainer:
    def __init__(self,):
        self.init_config()
        self.init()
        
    def init_config(self,):
        self.exp_name = 'stage1_vae'
        self.logs_dir = 'exp'
        self.epochs = 1000
        self.batch_size = 32
        self.in_dim = 3
        self.latent_dim = 1024
        self.lr = 0.0001
        self.weight_decay = 0
        self.scheduler_gamma = 0.95
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_dir = os.path.join(self.logs_dir, self.exp_name)
        self.sample_dir = os.path.join(self.exp_dir,'samples')
        # self.trainset_dir = os.path.join(os.getcwd(),'data', '7.5k', "train")
        self.trainset_dir = r"data/"
        self.valset_dir = r"data/"
        self.sample_interval = 100
        self.img_size = (224, 224)
        self.tb_log_dir = os.path.join(self.exp_dir,'tb_logs')
        self.workers = 8
        self.ckpt_save_interval = 2

        self.alpha = 0.00025
        self.beta = 100
        self.gamma = 4

        self.use_complie = False
        self.mixed_precision = False
        self.precision = torch.bfloat16 if self.mixed_precision else torch.float32
        

    def init(self):
        # shutil.rmtree(self.tb_log_dir, ignore_errors=True)
        shutil.rmtree(self.sample_dir, ignore_errors=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.tb_log_dir, exist_ok=True)

        # os.remove(os.path.join(self.sample_dir, '*'))
        self.trainset = ImageDataset(self.trainset_dir, is_train=True, im_size=self.img_size)
        self.train_loader = DataLoader(
                    self.trainset,
                    batch_size=self.batch_size, 
                    shuffle=True,
                    num_workers=self.workers,
                    pin_memory=True,
                    persistent_workers=True
                    )
        self.valset = ImageDataset(self.valset_dir, is_train=False, im_size=self.img_size)
        self.val_loader = DataLoader(
                        self.valset,
                        batch_size=self.batch_size, 
                        shuffle=True,
                        num_workers=self.workers,
                        pin_memory=True,
                        persistent_workers=True
                        )


        self.model = VAEModel(self.latent_dim)

        if self.use_complie:
            self.model = torch.compile(self.model)

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.perceptual_loss = LPIPS().to(self.device).eval()

        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.scheduler_gamma)

        if not self.try_load_checkpoint():
            self.epoch = 0
            self.global_step = 0
            self.global_val_step = 0

        self.tb_logger = SummaryWriter(log_dir=self.tb_log_dir)

        print(f"Number of parameters in model: {sum(p.numel() for p in self.model.parameters())/1000000:.2f} M")

    def try_load_checkpoint(self, )->None:
        ckpt_paths = glob.glob(os.path.join(self.exp_dir, "ckpts", f"*.pth"))
        if len (ckpt_paths) == 0:
            print("No checkpoint found, start from scratch")
            return False
        def get_epoch_step(s):
            s=s.replace('\\', os.sep).replace('/', os.sep)
            s=s.split(os.sep)[-1]
            match = re.search(r'ckpt_epoch_(\d+)_step_(\d+).pth', s)
            if match:
                epoch = int(match.group(1))
                # step = int(match.group(2))
                return epoch
            else:
                return s
        ckpt_paths.sort(reverse=True, key=get_epoch_step)
        self.load_checkpoint(ckpt_paths[0])
        print(f"Resume from epoch {self.epoch}, step {self.global_step}")

        return True
        
    def load_checkpoint(self, checkpoint_path:str)->None:
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        print(self.model.load_state_dict(ckpt['weights'], strict=False))
        self.optimizer.load_state_dict(ckpt['optimizer'])

        self.alpha = ckpt.get('alpha', self.alpha)
        self.beta = ckpt.get('beta', self.beta)
        self.gamma = ckpt.get('gamma', self.gamma)
        print(f"Loaded alpha: {self.alpha}, beta: {self.beta}, gamma: {self.gamma}")
        self.epoch = ckpt['epoch']
        self.global_step = ckpt['step']
        self.global_val_step = ckpt['val_step']



    def save_checkpoint(self,)->None:
        save_dir = os.path.join(self.exp_dir, "ckpts")
        os.makedirs(save_dir, exist_ok=True)

        torch.save({
            "exp_name": self.exp_name,
            'epoch': self.epoch,
            'step': self.global_step,
            'val_step': self.global_val_step,
            'weights': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, os.path.join(save_dir, f"ckpt_epoch_{self.epoch}_step_{self.global_step}.pth"))


    def save_weights(self)->None:
        weights={
            "exp_name": self.exp_name,
            "epoch": self.epoch,
            "step": self.global_step,
            "weights": self.model.state_dict()
        }
        save_dir = os.path.join(self.exp_dir,"weights")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(weights, os.path.join(save_dir, f"VAE_epoch_{self.epoch}_step_{self.global_step}.pth"))
    
    def run(self,):
        set_seed(1265)
        for epoch in range(self.epochs):
            self.epoch += 1
            self.train(self.epoch)
            self.scheduler.step()
            if self.epoch % self.ckpt_save_interval == 0:
                self.save_weights()
                self.save_checkpoint()
            self.val(self.epoch)
            


    
    def train(self, epoch):
        self.model.train()

        batch_idx = -1
        for (data) in self.train_loader:
            batch_idx += 1
            self.global_step += 1

            img = data["img"].to(self.device)
            # cond = data["label"].to(self.device)
            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            with torch.autocast(device_type=device_type, enabled=self.mixed_precision, dtype=self.precision):
                x_hat, mu, logvar = self.model(img)

                with torch.autocast(device_type=device_type, enabled=False):
                    recon_loss, kl_loss = self.model.compute_loss(x_hat.float(), img.float(), mu.float(), logvar.float())
                    feature_loss = self.perceptual_loss(x_hat.float(), img.float()).mean()
                    loss = self.alpha * kl_loss + self.beta * recon_loss+ self.gamma * feature_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()



            self.tb_logger.add_scalar('train/loss', loss, self.global_step)
            self.tb_logger.add_scalar('train/recon_loss', recon_loss, self.global_step)
            self.tb_logger.add_scalar('train/kl_loss', kl_loss, self.global_step)
            self.tb_logger.add_scalar('train/feature_loss', feature_loss, self.global_step)
            self.tb_logger.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            self.tb_logger.add_scalar('train/varaince', logvar.detach().exp().mean(), self.global_step)


            print(f'====> Epoch: {epoch} | Batch: {batch_idx+1}/{len(self.train_loader)} | loss: {loss:.8f}')

            if self.global_step % self.sample_interval == 0:
                # samples = self.save_samples(self.global_step)
                self.tb_logger.add_images('img/x_hat', x_hat.float().cpu().detach().numpy()[:4], self.global_step)
                self.tb_logger.add_images('img/x', img.float().cpu().detach().numpy()[:4], self.global_step)
                # self.tb_logger.add_images('img/samples', samples.cpu().detach().numpy()[:4], self.global_step)


                
        
    def val(self, epoch):
        self.model.eval()
        total_loss = 0

        total_recon_loss = 0
        total_kl_loss = 0
        total_feature_loss = 0

        
        with torch.no_grad():
            for batch_idx, (data) in enumerate(self.val_loader):
                self.global_val_step += 1

                img = data["img"].to(self.device)
                # cond = data["label"].to(self.device)
                # Train VAE

                device_type = "cuda" if "cuda" in str(self.device) else "cpu"
                with torch.autocast(device_type=device_type, enabled=self.mixed_precision, dtype=self.precision):
                    x_hat, mu, logvar= self.model(img)
 
                    with torch.autocast(device_type=device_type, enabled=False):
                        recon_loss, kl_loss = self.model.compute_loss(x_hat.float(), img.float(), mu.float(), logvar.float())
                        feature_loss = self.perceptual_loss(x_hat.float(), img.float()).mean()
                        loss = self.alpha * kl_loss + self.beta * recon_loss + self.gamma * feature_loss

                if self.global_val_step % self.sample_interval == 0:
                    # samples = self.save_samples(self.global_step)
                    self.tb_logger.add_images('val/x_hat', x_hat.float().cpu().detach().numpy()[:4], self.global_val_step)
                    self.tb_logger.add_images('val/x', img.float().cpu().detach().numpy()[:4], self.global_val_step)
                    # self.tb_logger.add_images('img/samples', samples.cpu().detach().numpy()[:4], self.global_step)


                print(f'====> Epoch: {epoch} | Batch: {batch_idx+1}/{ len(self.val_loader)} | loss: {loss.item():.8f}')


                total_loss+= loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_feature_loss += feature_loss.item()

        
        print(f'====> Test avrage total_g_loss: {total_loss/len(self.val_loader):.8f}')
        self.tb_logger.add_scalar('val/loss', total_loss/len(self.val_loader), self.global_val_step)
        self.tb_logger.add_scalar('val/recon_loss', total_recon_loss/len(self.val_loader), self.global_val_step)
        self.tb_logger.add_scalar('val/kl_loss', total_kl_loss/len(self.val_loader), self.global_val_step)
        self.tb_logger.add_scalar('val/feature_loss', total_feature_loss/len(self.val_loader), self.global_val_step)
        self.model.train()


    def save_samples(self, epoch):
        self.model.eval()
        sample = self.model.sample(self.batch_size, self.device)
        # self.tb_logger.add_images('sample', sample.view(self.batch_size, 3, *self.img_size), self.global_step)
        torchvision.utils.save_image(sample.view(self.batch_size, 3, *self.img_size), 
                                    '{}/sample_{}.png'.format(self.sample_dir, epoch), 
                                    nrow=int(math.sqrt(self.batch_size))) 
        self.model.train()

        return sample
    

        
if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()