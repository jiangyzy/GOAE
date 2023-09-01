import torch
import torch.nn as nn
from torchvision import transforms
from enum import Enum
from models.attention import CrossAttention
from models.afa import AFA
from models.swin_transformer import build_model
from training.triplane import TriPlaneGenerator
import dnnlib
import legacy
from torch_utils import misc

class ProgressiveStage(Enum):
    WTraining = 0
    Delta1Training = 1
    Delta2Training = 2
    Delta3Training = 3
    Delta4Training = 4
    Delta5Training = 5
    Delta6Training = 6
    Delta7Training = 7
    Delta8Training = 8
    Delta9Training = 9
    Delta10Training = 10
    Delta11Training = 11
    Delta12Training = 12
    Delta13Training = 13
    Inference = 14


def get_mlp_layer(in_dim, out_dim, mlp_layer=2):
    module_list = nn.ModuleList()
    for j in range(mlp_layer-1):
        module_list.append(nn.Linear(in_dim, in_dim))
        module_list.append(nn.LeakyReLU())
    module_list.append(nn.Linear(in_dim, out_dim)) 
    return nn.Sequential(*module_list)


class GOAEncoder(nn.Module):
    def __init__(self, swin_config, mlp_layer=2, ws_dim = 14, stage_list=[20000, 40000, 60000]):
        super(GOAEncoder, self).__init__()
        self.style_count = ws_dim       
        self.stage_list = stage_list
        self.stage_dict =   {'base':0, 'coarse':1, 'mid':2, 'fine':3} 
        self.stage = 3

## -------------------------------------------------- base w0 swin transformer -------------------------------------------
        self.swin_model = build_model(swin_config)
        
        self.mapper_base_spatial = get_mlp_layer(64, 1, mlp_layer)
        self.mapper_base_channel = get_mlp_layer(1024, 512, mlp_layer)

        self.maxpool_base = nn.AdaptiveMaxPool1d(1)

## -------------------------------------------------- w Query mapper coarse mid fine  1024*64 -> (4-1)*512 3*512 7*512 -------------------------------------------
        self.maxpool_query = nn.AdaptiveMaxPool1d(1)

        self.mapper_query_spatial_coarse = get_mlp_layer(64, 3, mlp_layer)
        self.mapper_query_channel_coarse = get_mlp_layer(1024, 512, mlp_layer)
        self.mapper_query_spatial_mid = get_mlp_layer(64, 3, mlp_layer)
        self.mapper_query_channel_mid = get_mlp_layer(1024, 512, mlp_layer)
        self.mapper_query_spatial_fine = get_mlp_layer(64, 7, mlp_layer)
        self.mapper_query_channel_fine = get_mlp_layer(1024, 512, mlp_layer)

## -------------------------------------------------- w KQ coarse mid fine mapper to 512 -------------------------
        self.mapper_coarse_channel = nn.Sequential(nn.Linear(512,512), nn.LeakyReLU())      
        self.mapper_mid_channel = nn.Sequential(nn.Linear(256,512), nn.LeakyReLU())         
        self.mapper_fine_channel = nn.Sequential(nn.Linear(128,256), nn.LeakyReLU(), nn.Linear(256,512), nn.LeakyReLU())   

        self.mapper_coarse_to_mid_spatial = nn.Sequential(nn.Linear(256,512), nn.LeakyReLU(), nn.Linear(512,1024), nn.LeakyReLU())
        self.mapper_mid_to_fine_spatial = nn.Sequential(nn.Linear(1024,2048), nn.LeakyReLU(), nn.Linear(2048,4096), nn.LeakyReLU())

## -------------------------------------------------- w KQ coarse mid fine Cross Attention -------------------------
        self.cross_att_coarse = CrossAttention(512, 4, 1024, batch_first=True)
        self.cross_att_mid = CrossAttention(512, 4, 1024, batch_first=True)
        self.cross_att_fine = CrossAttention(512, 4, 1024, batch_first=True)
        self.progressive_stage = ProgressiveStage.Inference


    def set_stage(self, iter):
        if iter > self.stage_list[-1]:
            self.stage = 3
        else:
            for i, stage_iter in enumerate(self.stage_list):
                if iter < stage_iter:
                    break
            self.stage = i
        
        print(f"change training stage to {self.stage}")


    def forward(self, x):
        B = x.shape[0]
        x_base, x_query, x_coarse, x_mid, x_fine= self.swin_model(x)

## ----------------------  base 
        ws_base_max = self.maxpool_base(x_base).transpose(1, 2)     
        ws_base_linear = self.mapper_base_spatial(x_base)     
        ws_base = self.mapper_base_channel(ws_base_linear.transpose(1, 2)  + ws_base_max )   

        ws_base = ws_base.repeat(1,14,1)     

        if self.stage == self.stage_dict['base']:
            ws = ws_base
            return ws, ws_base

## ------------------------ coarse mid fine ---  query    


        ws_query_max = self.maxpool_query(x_query).transpose(1, 2)  


        if self.stage >= self.stage_dict['coarse']:
            ws_query_linear_coarse = self.mapper_query_spatial_coarse(x_query)    
            ws_query_coarse = self.mapper_query_channel_coarse(ws_query_linear_coarse.transpose(1, 2)  + ws_query_max ) 

            if self.stage >= self.stage_dict['mid']:
                ws_query_linear_mid = self.mapper_query_spatial_mid(x_query)      
                ws_query_mid = self.mapper_query_channel_mid(ws_query_linear_mid.transpose(1, 2)  + ws_query_max )   

                if self.stage >= self.stage_dict['fine']:
                    ws_query_linear_fine = self.mapper_query_spatial_fine(x_query)  
                    ws_query_fine = self.mapper_query_channel_fine(ws_query_linear_fine.transpose(1, 2)  + ws_query_max )  


## -------------------------  carse, mid, fine -----  key-value 
        if self.stage >= self.stage_dict['coarse']:
            kv_coarse = self.mapper_coarse_channel(x_coarse)     

            if self.stage >= self.stage_dict['mid']:
                kv_mid = self.mapper_mid_channel(x_mid) + self.mapper_coarse_to_mid_spatial(kv_coarse.transpose(1, 2)).transpose(1, 2)    

                if self.stage >= self.stage_dict['fine']:
                    kv_fine = self.mapper_fine_channel(x_fine) + self.mapper_mid_to_fine_spatial(kv_mid.transpose(1, 2)).transpose(1, 2) 
        
        

## ------------------------- carse, mid, fine -----  Cross attention
        if self.stage >= self.stage_dict['coarse']:
            ws_coarse = self.cross_att_coarse(ws_query_coarse, kv_coarse )
            zero_1 = torch.zeros(B,1,512).to(ws_base.device)
            zero_2 = torch.zeros(B,10,512).to(ws_base.device)
            ws_delta = torch.cat([zero_1, ws_coarse, zero_2], dim=1)

            if self.stage >= self.stage_dict['mid']:
                ws_mid = self.cross_att_mid(ws_query_mid, kv_mid)
                zero_1 = torch.zeros(B,1,512).to(ws_base.device)
                zero_2 = torch.zeros(B,7,512).to(ws_base.device)
                ws_delta = torch.cat([zero_1, ws_coarse, ws_mid, zero_2], dim=1)


                if self.stage >= self.stage_dict['fine']:
                    ws_fine = self.cross_att_fine(ws_query_fine, kv_fine)
        
                    zero = torch.zeros(B,1,512).to(ws_base.device)

                    ws_delta = torch.cat([zero, ws_coarse, ws_mid, ws_fine], dim=1)

        ws = ws_base + ws_delta
        return ws, ws_base




class Net(nn.Module):
    def __init__(self, device, opts, swin_config) -> None:
        super(Net, self).__init__()

        self.decoder = self.set_generator(device, opts)
        self.encoder = self.set_encoder(device, opts, swin_config)

        self.start_from_latent_avg = opts.start_from_latent_avg
        if self.start_from_latent_avg:
            self.w_avg = self.decoder.backbone.mapping.w_avg[None, None, :].to(device)         # 1, 1, 512
        
        self.afa = self.set_afa(device, opts)

        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

        self.c_front = torch.tensor([1,0,0,0, 
                        0,-1,0,0,
                        0,0,-1,2.7, 
                        0,0,0,1, 
                        4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).float().to(device).reshape(1,-1)
        self.device = device
        self.gauss_kernel_0 = transforms.GaussianBlur(9, sigma=(1,2))  
        self.gauss_kernel_1 = transforms.GaussianBlur(9, sigma=(1,2))  



    def set_generator(self, device, opts):
        with dnnlib.util.open_url(opts.G_ckpt) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) 
        
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
        return G


    def set_encoder(self, device, opts, swin_config):
        E =  GOAEncoder(swin_config, mlp_layer=opts.mlp_layer, stage_list=[10000, 20000, 30000]).to(device)

        if opts.E_ckpt:
            E_ckpt = torch.load(opts.E_ckpt, map_location=device) 
            E.load_state_dict(E_ckpt)

        num_params = sum(param.numel() for param in E.parameters())
        print("Encoder parmeters number is :    ", num_params)
        return E


    def set_afa(self, device, opts):

        afa = AFA().to(device)

        if opts.AFA_ckpt:
            AFA_ckpt = torch.load(opts.AFA_ckpt, map_location=device)
            afa.load_state_dict(AFA_ckpt)

        num_params = sum(param.numel() for param in afa.parameters())
        print("AFA parmeters number is :    ", num_params)   

        return afa


    def forward(self, x, c, x_512, return_latents=False):    
        B = x.shape[0]

        ## GOAE Encoder
        rec_ws, _ = self.encoder(x)

        if self.start_from_latent_avg: 
            rec_ws = rec_ws + self.w_avg.repeat(B, 1, 1)

        triplane, triplane_x = self.decoder.synthesis(ws=rec_ws, c=c, train_forward_1=True, noise_mode='const')
        rec_img_dict_w = self.decoder.synthesis(ws=rec_ws, c=c, 
                                    triplane=triplane, triplane_x=triplane_x, train_forward_2=True, noise_mode='const')

        rec_img_w = rec_img_dict_w['image']

        ## AFA
        feature_map_adain, gamma, beta = self.afa(x_512, rec_img_w, triplane_x)

        rec_img_dict = self.decoder.synthesis(ws=rec_ws, c=c, 
                                    triplane=triplane, triplane_x=feature_map_adain, train_forward_2=True, noise_mode='const')  

        ## Occlusion-Aware mix Fusion
        mix_triplane, mask = self.mix_fusion(B, rec_img_dict, rec_img_dict_w)

        rec_img_dict['mix_triplane'] = mix_triplane

        rec_img_dict['mask'] = mask


        if return_latents:
            rec_img_dict['rec_ws'] = rec_ws

        return rec_img_dict, rec_img_dict_w
    
    def mix_fusion(self, bsz, rec_img_dict, rec_img_dict_w):

        with torch.no_grad():
            ray_origins = rec_img_dict['ray_origins']            # B 128*128  3 
            ray_directions = rec_img_dict['ray_directions']     # B 128*128  3 
            depth = rec_img_dict['image_depth'].flatten(start_dim=1).reshape(bsz, -1, 1).detach()         # B 128*128  1    
            xyz = ray_origins + depth * ray_directions          # surface xyz       B 128*128  3 

            x = (xyz[:,:,0] + 0.5)*256
            y = (xyz[:,:,1] + 0.5)*256
            x = torch.clamp(x, min=0, max=255)
            y = torch.clamp(y, min=0, max=255)
            x = x.long()
            y = y.long()

            mask = torch.zeros((bsz,1,256,256))
            for j in range(bsz):
                ## visulable range
                mask[j, :, y[j], x[j]] = 1
                mask[j] = self.gauss_kernel_0(mask[j])
                mask[j] = torch.where(mask[j] > 0.35, 1, 0).float()
                mask[j] = self.gauss_kernel_1(mask[j])
        mask = mask.to(self.device)

        f_triplane = rec_img_dict['triplane']
        w_triplane = rec_img_dict_w['triplane']  
        mix_triplane = torch.zeros_like(f_triplane)   

        ## only fusion on plane_xy 
        mix_triplane[:,0,:,:,:] = mask * f_triplane[:,0,:,:,:] + (1-mask) * w_triplane[:,0,:,:,:]
        mix_triplane[:,1:,:,:,:] = f_triplane[:,1:,:,:,:]       

        ## fusion on all tri_plane
        # mix_triplane = mask * f_triplane + (1-mask) * w_triplane

        return mix_triplane, mask


    def edit(self, ws_rec, edit_ws, x_512, c):  
        B = ws_rec.shape[0]
        
        ## W + space
        triplane_rec, triplane_x_rec = self.decoder.synthesis(ws=ws_rec, c=c, train_forward_1=True, noise_mode='const')  
        rec_img_dict_w_rec = self.decoder.synthesis(ws=ws_rec, c=c, 
                                    triplane=triplane_rec, triplane_x=triplane_x_rec, train_forward_2=True, noise_mode='const')    

        triplane_edit, triplane_x_edit = self.decoder.synthesis(ws=edit_ws, c=c, train_forward_1=True, noise_mode='const')  
        rec_img_dict_w_edit = self.decoder.synthesis(ws=edit_ws, c=c, 
                                    triplane=triplane_edit, triplane_x=triplane_x_edit, train_forward_2=True, noise_mode='const')            

        img_rec_w = rec_img_dict_w_rec['image']
        img_edit_dict_w = rec_img_dict_w_edit['image']

        ## F space
        w_triplane = rec_img_dict_w_edit['triplane']
        feature_map_adain_rec, gamma_rec, beta_rec = self.afa(x_512, img_rec_w, triplane_x_rec)
        feature_map_adain = feature_map_adain_rec + (triplane_x_edit - triplane_x_rec)


        img_dict = self.decoder.synthesis(ws=ws_rec, c=c, 
                                    triplane=triplane_edit, triplane_x=feature_map_adain, train_forward_2=True, noise_mode='const')

        ## Occlusion-Aware mix Fusion
        mix_triplane, mask = self.mix_fusion(B, img_dict, rec_img_dict_w_edit)

        img_edit_dict = self.decoder.synthesis(ws=edit_ws, c=c,
                                    triplane=mix_triplane, forward_triplane=True, noise_mode='const') 
        
        img_edit_dict['mix_triplane'] = mix_triplane

        return  img_edit_dict, img_edit_dict_w