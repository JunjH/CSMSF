import numpy as np
from PIL import Image
from data_transform import *
import pandas as pd

class depthDataset(Dataset):
    def __init__(self, path, file, transform=None, is_train=True):
        self.path = path
        self.file = file
        self.frame = pd.read_csv(self.path + self.file, header=None)
        self.transform = transform

    def __getitem__(self, idx):
        path_img = self.path + self.frame.loc[idx,0]
        path_thr = path_img.replace('rgb','thr')
        path_nir = path_img.replace('rgb','nir')

        path_lidar_img_sd = path_img.replace('sync_data','proj_depth').replace('img_left','depth')
        path_lidar_img_gt = path_img.replace('sync_data','proj_depth').replace('img_left','depth_filtered')
        path_lidar_thr_sd = path_thr.replace('sync_data','proj_depth').replace('img_left','depth')
        path_lidar_thr_gt = path_thr.replace('sync_data','proj_depth').replace('img_left','depth_filtered')
        path_lidar_nir_sd = path_nir.replace('sync_data','proj_depth').replace('img_left','depth')
        path_lidar_nir_gt = path_nir.replace('sync_data','proj_depth').replace('img_left','depth_filtered')

        img = Image.open(path_img).resize((640,256))
        thr = Image.open(path_thr).resize((640,256))
        nir = Image.open(path_nir).resize((640,256))

        lidar_img_sd = Image.open(path_lidar_img_sd).resize((640,256),Image.NEAREST)
        lidar_img_gt = Image.open(path_lidar_img_gt).resize((640,256),Image.NEAREST)
        lidar_thr_sd = Image.open(path_lidar_thr_sd)
        lidar_thr_gt = Image.open(path_lidar_thr_gt)
        lidar_nir_sd = Image.open(path_lidar_nir_sd).resize((640,256),Image.NEAREST)
        lidar_nir_gt = Image.open(path_lidar_nir_gt).resize((640,256),Image.NEAREST)

        sample = {'img': img, 'lidar_img_sd': lidar_img_sd, 'lidar_img_gt': lidar_img_gt, 'thr': thr, 'lidar_thr_sd': lidar_thr_sd, 'lidar_thr_gt': lidar_thr_gt}


        if self.transform:
            sample = self.transform(sample)
            
        return sample


    def __len__(self):
        return len(self.frame)


def getTrainingData(batch_size,path,file):

    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_training = depthDataset(path,file,
                                        transform=transforms.Compose([
                                            RandomHorizontalFlip(),                        
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]), is_train=True)

    dataloader_training = DataLoader(transformed_training, batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return dataloader_training


def getTestingData(batch_size,path,file):

    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_testing = depthDataset(path,file,
                                        transform=transforms.Compose([
                                            ToTensor(),    
                                            ColorJitter(
                                                brightness=0,
                                                contrast=0,
                                                saturation=0,
                                            ),                                 
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]), is_train=False)

    dataloader_testing = DataLoader(transformed_testing, batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return dataloader_testing




def Raw2Celsius(Raw):
    R = 380747
    B = 1428
    F = 1
    O = -88.539
    Celsius = B / np.log(R / (Raw - O) + F) - 273.15;
    return Celsius


def set_extrinsics(path) :
    # extrinsics matries are all same across the sequences, thus use the same values
    calib_path = osp.join(path, "calib.npy")
    calib = np.load(calib_path, allow_pickle=True).item()

    ext_NIR2THR = np.concatenate([calib['R_nir2thr'], calib['T_nir2thr']*0.001], axis=1) # mm -> m scale conversion.
    ext_NIR2RGB = np.concatenate([calib['R_nir2rgb'], calib['T_nir2rgb']*0.001], axis=1)

    ext_THR2NIR = np.linalg.inv(np.concatenate([ext_NIR2THR, [[0,0,0,1]]],axis=0))
    ext_THR2RGB = np.matmul(np.concatenate([ext_NIR2RGB, [[0,0,0, 1]]],axis=0), ext_THR2NIR)

    ext_RGB2NIR = np.linalg.inv(np.concatenate([ext_NIR2RGB, [[0,0,0,1]]],axis=0))
    ext_RGB2THR = np.linalg.inv(ext_THR2RGB)

    extrinsics = {}
    extrinsics["NIR2THR"] = torch.as_tensor(ext_NIR2THR)
    extrinsics["NIR2RGB"] = torch.as_tensor(ext_NIR2RGB)

    extrinsics["THR2NIR"] = torch.as_tensor(ext_THR2NIR[0:3,:])
    extrinsics["THR2RGB"] = torch.as_tensor(ext_THR2RGB[0:3,:])

    extrinsics["RGB2NIR"] = torch.as_tensor(ext_RGB2NIR[0:3,:])
    extrinsics["RGB2THR"] = torch.as_tensor(ext_RGB2THR[0:3,:])
    return extrinsics


def set_intrinsics() :
    intrinsics = {}
    intrinsics["rgb"] = calib['K_rgbL'].astype(np.float32)
    intrinsics["nir"] = calib['K_nirL'].astype(np.float32)
    intrinsics["thr"] = calib['K_thrL'].astype(np.float32)

    return intrinsics

if __name__ == '__main__':
    main()
