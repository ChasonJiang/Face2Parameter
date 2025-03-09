import json
import os
import random
import cv2
import numpy as np
import torch
import torch.utils
import torchvision
from src.models.MLP.MLP import MLP
from src.models.VAE.model import VAEModel
from src.face_data_utils.FaceCrop import FaceCrop
from src.face_data_utils.utils import FaceData
import torch.nn.functional as F
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

torch.manual_seed(123)
torch.cuda.random.manual_seed(123)
random.seed(123)

class Extractor(object):
    def __init__(self,):
        self.initConfig()
        self.init()


    def initConfig(self):
        # self.num_dim = 59 -18
        self.in_dim = 2048
        self.out_dim = 205
        self.hidden_dim = 1024
        self.num_layers = 8
        self.device = torch.device("cuda:0")
        self.mlp_weight_path = r"checkpoints\mlp_8_1024_0.0001_100\epoch 30.pth"
        self.vae_weight_path = r"checkpoints\vae\VAE_110k_epoch_30_step_189450.pth"
        self.im_size = None

    def init(self):
        self.model = MLP(self.in_dim, self.out_dim, self.hidden_dim, num_layers=self.num_layers)
        # self.load_model(self.mlp_weight_path,self.model)
        self.model.load_state_dict(torch.load(self.mlp_weight_path, map_location="cpu"))
        self.model.to(self.device)
        self.model.eval()

        self.vae = VAEModel(self.in_dim//2)
        self.vae.load_state_dict(torch.load(self.vae_weight_path, map_location="cpu")["weights"])
        self.vae.to(self.device)
        self.vae.eval()

        self.face_crop = FaceCrop()
        
        
    def extract(self, image_path:str, save_dir:str, template_path:str, use_face_detector=True):
        os.makedirs(save_dir, exist_ok=True)
        face = self.process_image(image_path, use_face_detector)
        face = face.unsqueeze(0).to(self.device)

        with torch.no_grad():
            mu, logvar = self.vae.encoder(face)
            latent_vec = torch.cat([mu, logvar], dim=1)
            output=self.model(latent_vec)
            output = output.squeeze(0).cpu().numpy()
            z = self.vae.reparameterize(mu, logvar)
            face_hat:torch.Tensor = self.vae.decode(z)
        
        face_data = FaceData(template_path)
        face_data.set_from_vector(output, is_simplify=True, without_right=True, denormalize=True, use_gaussian=False)
        
        face_image = F.pad(face_hat.squeeze(0).permute(1,2,0)*255,[ 0,0, 0,0, 44, 45,], mode='constant', value=0).cpu().numpy().astype(np.uint8)
        face_data.set_image(face_image)
        
        save_path = os.path.join(save_dir, os.path.basename(image_path).split('.')[0] + '.png')
        face_data.save(save_path)

        return output.tolist()

    def load_model(self, load_path, model, strict=True):
        load_net = torch.load(load_path)
        model.load_state_dict(load_net, strict=strict)
        model.eval()



    def process_image(self, file_path:str, use_face_detector=True, im_size=(224,224)):
        img=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if use_face_detector:
            faces = self.face_crop.crop(img)
            if len(faces)==0:
                raise RuntimeError("No face detected in the image")
            if len(faces)>1:
                raise RuntimeError("More than one face detected in the image")
            face = faces[0]
        else:
            face = img
        if face.shape[0]!= im_size[0] or face.shape[1]!= im_size[1]:
            face = cv2.resize(face, im_size, interpolation = cv2.INTER_AREA)
        # HWC to CHW
        face_np=np.transpose(face, (2,0,1))
        face_tensor = torch.from_numpy(np.ascontiguousarray(face_np)).to(self.device).float().div(255)
        # torchvision.utils.save_image(face_tensor, file_path.replace('.png','_cropped.png'))
        return face_tensor
    
if __name__ =="__main__":
    # Step 1, Create an Extractor instance
    extractor = Extractor()
    # Step 2, Extract the face data from image to json file

    image_names = [
                    ("my.png", "template.png"), 
                    ("yuechan.png", "yuechan_template.png"), 
                    ("xw.png", "xw_template.png"), 
                    ("hler2.jpg", "hler_template.png"),
                    ("lyf.jpg", "template.png"), 
                    ("tt.jpg", "template.png"), 
                ]
    
    for image_name, template_name in image_names:
        image_path = r"test\{}".format(image_name)
        save_dir = "outputs"
        template_path = "test/{}".format(template_name)
        # image_path = r"test\my.png"
        # save_dir = "outputs"
        # template_path = "test/template.png"
        use_face_detector = True
        data=extractor.extract(image_path=image_path, save_dir=save_dir, template_path=template_path, use_face_detector=use_face_detector)
        # [Optional] Step 3, Print face data to the console
        print(data)