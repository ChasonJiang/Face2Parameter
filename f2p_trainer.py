import glob
import os
import random
import re
import time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataset.datasets import FeatureDataset
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter

from src.models.MLP.MLP import MLP

# from sklearn.metrics.pairwise import cosine_similarity

# torch.manual_seed(123)
# torch.cuda.random.manual_seed(123)
# random.seed(123)

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


class Trainer(object):
    def __init__(self,):
        self.initConfig()
        self.init()

    def initConfig(self):
        # string. The root directory of the dataset
        self.seed = 123
        self.exp_name = "stage2_mlp_x4_5e6_0.0"
        self.trainset_dir = r"data/"
        self.valset_dir = r"data/"
        self.batch_size = 32
        self.in_dim = 2048
        self.out_dim = 205
        self.hidden_dim = 1024
        self.num_layers = 4 # or 8
        # int. Number of epoch to learn
        self.num_epoch = 100
        # float. learning rate
        self.lr = 0.000005 # or 1e4
        self.aug_prob = 0.0
        # "cpu" or "cuda"
        self.device = torch.device("cuda")
        self.exp_dir = os.path.join("./exp", self.exp_name)
        # int. Frequency of ckpt saving
        self.ckpt_save_interval = 1
        self.val_interval = 1
        # string. Saving path of tensorboard log
        self.tb_log_save_path = os.path.join(self.exp_dir, "tb_logs")


    def init(self):
        os.makedirs(self.tb_log_save_path,exist_ok=True)
        # self.remove_file_in_dir(self.tb_log_save_path)
        # time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        log_name = f"{self.num_layers}_{self.hidden_dim}_{self.lr}_{self.num_epoch}"
        self.tb_log_save_path = os.path.join(self.tb_log_save_path, log_name)
        os.makedirs(self.tb_log_save_path,exist_ok=True)
        self.tb_logger = SummaryWriter(log_dir=self.tb_log_save_path,)
        self.dataset = FeatureDataset(self.trainset_dir, is_train=True, aug_prob=self.aug_prob, seed=self.seed)
        self.dataLoader = DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True,num_workers=8, pin_memory=True, persistent_workers=True)
        self.val_dataset = FeatureDataset(self.valset_dir, is_train=False, aug_prob=self.aug_prob, seed=self.seed)
        self.val_dataLoader = DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=True,num_workers=8, pin_memory=True, persistent_workers=True)

        self.model = MLP(self.in_dim, self.out_dim, self.hidden_dim, self.num_layers)
        self.model=self.model.to(self.device)
        self.lossfunc  = MSELoss(reduction="mean").to(self.device)
        self.optim =torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # self.scheduler=torch.optim.lr_scheduler.StepLR(self.optim, 1, gamma=0.2, last_epoch=-1)
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optim,self.num_epoch)
        # self.evaluator = Evaluator(False)
    
        if not self.try_load_checkpoint():
            self.epoch = 0
            self.global_step = 0
            self.global_val_step = 0


    def remove_file_in_dir(self,dir):
        for file in glob.glob(os.path.join(dir,"*")):  
            os.remove(file)  
            print("Deleted " + str(file)) 


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
        self.optim.load_state_dict(ckpt['optimizer'])

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
            'optimizer': self.optim.state_dict(),
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
        torch.save(weights, os.path.join(save_dir, f"MLP_epoch_{self.epoch}_step_{self.global_step}.pth"))
    





    def train(self,):
        set_seed(self.seed)
        self.model.train()
        for i in range(self.epoch, self.num_epoch-self.epoch):
            self.epoch = i+1
            for idx, data in enumerate(self.dataLoader):
                self.global_step+=1
                self.optim.zero_grad()
                feat:torch.Tensor = data["feat"]
                labels:torch.Tensor = data["label"]
                feat=feat.to(self.device)
                labels = labels.to(self.device)
                output:torch.Tensor=self.model(feat)
                
                # loss=self.lossfunc(output,labels)

                # loss_whole = self.lossfunc(output[:,:5],labels[:,:5])
                # loss_chin = self.lossfunc(output[:,5:13],labels[:,5:13])
                # loss_cheek = self.lossfunc(output[:,13:19],labels[:,13:19])
                # loss_eye = self.lossfunc(output[:,19:32],labels[:,19:32])
                # loss_nose = self.lossfunc(output[:,32:47],labels[:,32:47])
                # loss_mouth = self.lossfunc(output[:,47:54],labels[:,47:54])
                # loss_amb = self.lossfunc(output[:,54:],labels[:,54:])
                # loss = loss_whole + loss_chin + loss_cheek + loss_eye + loss_nose + loss_mouth + 6*loss_amb
                
                

                loss1 = self.lossfunc(output[:,:54],labels[:,:54])
                loss2 = self.lossfunc(output[:,54:],labels[:,54:])
                loss = loss1 + loss2


                # old   loss1+loss2  
                # old1  loss(mean)
                # old2  loss = loss_whole + loss_chin + loss_cheek + loss_eye + loss_nose + loss_mouth + 6*loss_amb
                # old3  loss(sum)

                loss.backward()
                self.optim.step()

                
                lr = self.get_current_learning_rate()[0]
                similarity=torch.nn.functional.cosine_similarity(output.detach(),labels.detach(),dim=1)
                similarity = torch.mean(similarity)
                distance = torch.nn.functional.pairwise_distance(output.detach(),labels.detach(),p=2).mean()
                self.tb_logger.add_scalar("loss",loss, self.global_step)
                self.tb_logger.add_scalar("lr",lr,self.global_step)
                self.tb_logger.add_scalar("cosine similarity",similarity,self.global_step)
                self.tb_logger.add_scalar("distance",distance,self.global_step)
                
                

                print(f"epoch: {self.epoch} | batch: {self.global_step} | loss: {loss:.6f} | lr: {lr} | distance: {distance:.3f} | cosine similarity: {similarity.cpu().numpy():.3f}")


            self.scheduler.step()
            if (self.global_step) % self.val_interval == 0:
                self.val()
            if (self.global_step) % self.ckpt_save_interval == 0:
                self.save_weights()
                self.save_checkpoint()
            
        self.save_weights()
        self.save_checkpoint()
        
    def val(self,):
        self.global_val_step+=1
        avg_loss = 0
        avg_similarity = 0
        avg_distance = 0
        self.model.eval()
        with torch.no_grad():
            for idx, data in enumerate(self.dataLoader):
                feat:torch.Tensor = data["feat"]
                labels:torch.Tensor = data["label"]
                feat=feat.to(self.device)
                labels = labels.to(self.device)
                output:torch.Tensor=self.model(feat)
                
                # loss=self.lossfunc(output,labels)
                loss1 = self.lossfunc(output[:,:54],labels[:,:54])
                loss2 = self.lossfunc(output[:,54:],labels[:,54:])
                loss = loss1 + loss2
                
                similarity=torch.nn.functional.cosine_similarity(output.detach(),labels.detach(),dim=1)
                similarity = torch.mean(similarity)
                distance = torch.nn.functional.pairwise_distance(output.detach(),labels.detach(),p=2).mean()
                
                avg_loss += loss.item()
                avg_similarity += similarity.item()
                avg_distance += distance.item()

        avg_loss /= len(self.dataLoader)
        avg_similarity /= len(self.dataLoader)
        avg_distance /= len(self.dataLoader)
        print(f"val step: {self.global_val_step} | val loss: {avg_loss:.6f} | val distance: {avg_distance:.3f} | val cosine similarity: {avg_similarity:.3f}")

        self.tb_logger.add_scalar("val avg loss",avg_loss,self.global_val_step)
        self.tb_logger.add_scalar("val avg similarity",avg_similarity,self.global_val_step)
        self.tb_logger.add_scalar("val avg distance",avg_distance,self.global_val_step)
        

    def get_current_learning_rate(self):
        lr_l = []
        for param_group in self.optim.param_groups:
            lr_l.append(param_group['lr'])
        return lr_l
    


if __name__ =="__main__":
    trainer = Trainer()
    trainer.train()