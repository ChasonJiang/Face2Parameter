import numpy as np
import torch
import torchvision
import onnxruntime
from src.models.VAE.model import Encoder

from src.models.MLP.MLP import MLP



class Model(torch.nn.Module):
    def __init__(self, input_dim:int, output_dim:int, hidden_dim:int=1024, num_layers:int=8):
        super(Model, self).__init__()
        self.encoder = Encoder(input_dim//2)
        self.mlp = MLP(input_dim, output_dim, hidden_dim, num_layers=num_layers)
    
    def forward(self, x:torch.Tensor):
        mu, logvar = self.encoder(x)
        latent_vec = torch.cat([mu, logvar], dim=1)
        o = self.mlp(latent_vec)
        return o
    
    def load_weights(self, vae_weight_path:str, mlp_weight_path:str):
        vae_ckpt = torch.load(vae_weight_path, map_location="cpu")["weights"]
        encoder_weights = {k.replace("encoder.", ""):v for k,v in vae_ckpt.items() if "encoder" in k}
        self.encoder.load_state_dict(encoder_weights)
        mlp_weights = torch.load(mlp_weight_path, map_location="cpu")["weights"]
        self.mlp.load_state_dict(mlp_weights)




        
def onnx_export(vae_weight_path:str, 
                mlp_weight_path:str, 
                save_path:str,
                input_dim:int=2048,
                output_dim:int=205,
                hidden_dim:int=1024,
                num_layers:int=8
                ):
    model = Model(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    model.load_weights(vae_weight_path, mlp_weight_path)
    model.eval()

    image = torch.zeros([1,3,224,224])

    torch.onnx.export(model,
                    (image,),
                    save_path,
                    input_names=["image"],
                    output_names=["vector"],
                    )


if __name__ =="__main__":
    vae_weight_path = r"exp\stage1_vae\weights\VAE_epoch_18_step_133974.pth"
    mlp_weight_path = r"exp\stage2_mlp_x4_5e6_0.0\weights\MLP_epoch_20_step_44020.pth"
    save_path = r"outputs/mlp_8_1024_1e4_100.onnx"

    onnx_export(vae_weight_path, mlp_weight_path, save_path, num_layers=4)
    
    ort_session = onnxruntime.InferenceSession(save_path)
    image = np.zeros([1,3,224,224],dtype=np.float32)
    output=ort_session.run(['vector'], {'image': image})
    print(output)
    print(output[0].shape)