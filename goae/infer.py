import os
import datetime
import dnnlib
import numpy as np
import torch
import torchvision
from configs.infer_config import get_parser
from configs.swin_config import get_config
from training.goae import Net
from tqdm import tqdm
from camera_utils import LookAtPoseSampler
from gen_video import gen_interp_video
from gen_shape import gen_mesh




def get_pose(cam_pivot, intrinsics, yaw=None, pitch=None, yaw_range=0.35, pitch_range=0.15, cam_radius=2.7):
    
    if yaw is None:
        yaw = np.random.uniform(-yaw_range, yaw_range)
    if pitch is None:
            pitch = np.random.uniform(-pitch_range, pitch_range)

    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw, np.pi/2 + pitch, cam_pivot, radius=cam_radius, device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).reshape(1,-1)
    return c


def build_dataloader(data_path, batch=1, pin_memory=True, prefetch_factor=2):
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset_face.CameraLabeledDataset', path=data_path, 
                    use_labels=True, max_size=None, xflip=False,resolution=256, use_512 = True)
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    
    return dataloader, dataset


@torch.no_grad()
def infer_main(opts, device, now):

    ## camera parameters
    cam_pivot = torch.tensor([0, 0, 0.2], device=device)
    intrinsics = torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1], device=device)
    face_pool = torch.nn.AdaptiveAvgPool2d((256,256))

    ## build model
    swin_config = get_config(opts)
    net = Net(device, opts, swin_config)
    net.eval()


    ## build data
    dataloader, dataset = build_dataloader(data_path=opts.data, batch=opts.batch)

    ## main loop
    for data in tqdm(dataloader):
        real_img, real_label, real_img_512, img_name = data

        real_img = torch.tensor(real_img).to(device).to(torch.float32) / 127.5 - 1.
        real_label = torch.tensor(real_label).to(device)
        real_img_512 = torch.tensor(real_img_512).to(device).to(torch.float32) / 127.5 - 1.  


        rec_img_dict, rec_img_dict_w = net(real_img, real_label, real_img_512, return_latents=True)

        rec_img = face_pool(rec_img_dict['image'])
        mix_triplane = rec_img_dict['mix_triplane']
        rec_ws = rec_img_dict['rec_ws']

        save_dir = os.path.join(opts.outdir, opts.data.split("/")[-1], img_name[0].split(".")[0])
        os.makedirs(save_dir, exist_ok=True)
        torchvision.utils.save_image(torch.cat([real_img, rec_img]), os.path.join(save_dir, f'rec_img.jpg'), 
                                        padding=0, normalize=True, range=(-1,1))

        if opts.multi_view:
            imgs_multi_view = []
            coef = [ 1 , 0, -1 ]
            for j in range(3):
                yaw =  coef[j] * np.pi*25/360
                pitch = 0
                c = get_pose(cam_pivot=cam_pivot, intrinsics=intrinsics, yaw=yaw, pitch=pitch)

                img_dict_novel_view = net.decoder.synthesis(ws=rec_ws, c=c, triplane=mix_triplane, forward_triplane=True, noise_mode='const') 
                img_novel_view= face_pool(img_dict_novel_view["image"])
                imgs_multi_view.append(img_novel_view)

            torchvision.utils.save_image(torch.cat(imgs_multi_view), os.path.join(save_dir, f'multi_view.jpg'), 
                                        padding=0, normalize=True, range=(-1,1))

        if opts.video:
            video_path = os.path.join(save_dir, 'video.mp4')
            camera_lookat_point = torch.tensor(net.decoder.rendering_kwargs['avg_camera_pivot'], device=device)
            gen_interp_video(net.decoder, rec_ws, triplane=mix_triplane, 
                                mp4=video_path, image_mode='image', device=device, w_frames=opts.w_frames)

        if opts.shape:
            mesh_path = os.path.join(save_dir, 'mesh.mrc')
            gen_mesh(G = net.decoder, ws = None, triplane=mix_triplane, save_path=mesh_path, device=device) 


if __name__=="__main__":
    parser = get_parser()
    opts = parser.parse_args()

    print("="*50, "Using CUDA: " + opts.cuda, "="*50)
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.cuda
    device = torch.device('cuda:' + opts.cuda)

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    infer_main(opts, device, now)