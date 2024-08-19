import os
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
import time
import cv2
import openslide
import h5py
import numpy as np
from PIL import Image
from Register import Registers
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_image
from tqdm.autonotebook import tqdm
from torchsummary import summary
from model_bbdm.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model_bbdm.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from .utils_clam import print_network, collate_features,save_hdf5

@Registers.runners.register_with_name('BBDMRunner')
class BBDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmnet = BrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "LBBDM":
            bbdmnet = LatentBrownianBridgeModel(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet

    def load_model_from_checkpoint(self):
        states = None
        if self.config.model.only_load_latent_mean_std:
            if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
                states = torch.load(self.config.model.model_load_path, map_location='cpu')
        else:
            states = super().load_model_from_checkpoint()

        if self.config.model.normalize_latent:
            if states is not None:
                self.net.ori_latent_mean = states['ori_latent_mean'].to(self.config.training.device[0])
                self.net.ori_latent_std = states['ori_latent_std'].to(self.config.training.device[0])
                self.net.cond_latent_mean = states['cond_latent_mean'].to(self.config.training.device[0])
                self.net.cond_latent_std = states['cond_latent_std'].to(self.config.training.device[0])
            else:
                if self.config.args.train:
                    self.get_latent_mean_std()

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               verbose=True,
                                                               threshold_mode='rel',
                                                               **vars(config.model.BB.lr_scheduler)
)
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        if self.config.model.normalize_latent:
            if self.config.training.use_DDP:
                model_states['ori_latent_mean'] = self.net.module.ori_latent_mean
                model_states['ori_latent_std'] = self.net.module.ori_latent_std
                model_states['cond_latent_mean'] = self.net.module.cond_latent_mean
                model_states['cond_latent_std'] = self.net.module.cond_latent_std
            else:
                model_states['ori_latent_mean'] = self.net.ori_latent_mean
                model_states['ori_latent_std'] = self.net.ori_latent_std
                model_states['cond_latent_mean'] = self.net.cond_latent_mean
                model_states['cond_latent_std'] = self.net.cond_latent_std
        return model_states, optimizer_scheduler_states

    def get_latent_mean_std(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.data.train.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None
        max_batch_num = 30000 // self.config.data.train.batch_size

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None):
            (x, x_name), (x_cond, x_cond_name) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_mean = x_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            x_cond_mean = x_cond_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_cond_mean = x_cond_mean if total_cond_mean is None else x_cond_mean + total_cond_mean
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, total_ori_var=None, total_cond_var=None):
            (x, x_name), (x_cond, x_cond_name) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((x_cond_latent - cond_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            return total_ori_var, total_cond_var

        print(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        print(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)
            # break

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        print(self.net.ori_latent_mean)
        print(self.net.ori_latent_std)
        print(self.net.cond_latent_mean)
        print(self.net.cond_latent_std)

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        (x, x_name), (x_cond, x_cond_name) = batch
        x = x.to(self.config.training.device[0])
        x_cond = x_cond.to(self.config.training.device[0])

        loss, additional_info = net(x, x_cond)
        if write:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
            if additional_info.__contains__('recloss_noise'):
                self.writer.add_scalar(f'recloss_noise/{stage}', additional_info['recloss_noise'], step)
            if additional_info.__contains__('recloss_xy'):
                self.writer.add_scalar(f'recloss_xy/{stage}', additional_info['recloss_xy'], step)
        return loss

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        reverse_sample_path = make_dir(os.path.join(sample_path, 'reverse_sample'))
        reverse_one_step_path = make_dir(os.path.join(sample_path, 'reverse_one_step_samples'))

        (x, x_name), (x_cond, x_cond_name) = batch

        batch_size = x.shape[0] if x.shape[0] < 4 else 4

        x = x[0:batch_size].to(self.config.training.device[0])
        x_cond = x_cond[0:batch_size].to(self.config.training.device[0])

        grid_size = 4

        # samples, one_step_samples = net.sample(x_cond,
        #                                        clip_denoised=self.config.testing.clip_denoised,
        #                                        sample_mid_step=True)
        # self.save_images(samples, reverse_sample_path, grid_size, save_interval=200,
        #                  writer_tag=f'{stage}_sample' if stage != 'test' else None)
        #
        # self.save_images(one_step_samples, reverse_one_step_path, grid_size, save_interval=200,
        #                  writer_tag=f'{stage}_one_step_sample' if stage != 'test' else None)
        #
        # sample = samples[-1]
        sample = net.sample(x_cond, clip_denoised=self.config.testing.clip_denoised).to('cpu')
        image_grid = get_image_grid(sample, grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'skip_sample.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_skip_sample', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x_cond.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'condition.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_condition', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'ground_truth.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_ground_truth', image_grid, self.global_step, dataformats='HWC')
    def compute_w_loader(self, file_path, output_path, wsi, model,
         batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
        custom_downsample=1, target_patch_size=-1):
        """
        args:
            file_path: directory of bag (.h5 file)
            output_path: directory to save computed features (.h5 file)
            model: pytorch model
            batch_size: batch_size for computing features in batches
            verbose: level of feedback
            pretrained: use weights pretrained on imagenet
            custom_downsample: custom defined downscale factor of image patches
            target_patch_size: custom defined, rescaled image size before embedding
        """
        device = self.config.training.device[0]
        dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
            custom_downsample=custom_downsample, target_patch_size=target_patch_size)
        x, y = dataset[0]
        kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
        loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

        if verbose > 0:
            print('processing {}: total of {} batches'.format(file_path,len(loader)))
        def restore_features(cur_features):
            f_1.append(cur_features[0].cpu().numpy())
            f_2.append(cur_features[1].cpu().numpy())
            f_3.append(cur_features[2].cpu().numpy())
            f_4.append(cur_features[3].cpu().numpy())
            f_5.append(cur_features[4].cpu().numpy())
            f_6.append(cur_features[5].cpu().numpy())
            f_7.append(cur_features[6].cpu().numpy())
            f_8.append(cur_features[7].cpu().numpy())
            f_9.append(cur_features[8].cpu().numpy())
            f_10.append(cur_features[9].cpu().numpy())
            return f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10
        def concat_features(f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10):
            f_1 = np.concatenate(f_1)
            f_2 = np.concatenate(f_2)
            f_3 = np.concatenate(f_3)
            f_4 = np.concatenate(f_4)
            f_5 = np.concatenate(f_5)
            f_6 = np.concatenate(f_6)
            f_7 = np.concatenate(f_7)
            f_8 = np.concatenate(f_8)
            f_9 = np.concatenate(f_9)
            f_10 = np.concatenate(f_10)
            return f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10
        mode = 'w'
        for count, (batch, coords) in enumerate(loader):
            with torch.no_grad():    
                if count % print_every == 0:
                    print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
                batch = batch.to(device, non_blocking=True)
                '''
                print(batch.shape)
                print(batch[0].max(),batch.min())
                for i in range(batch.shape[0]):
                    visual = batch[i]*127.5+127.5
                    visual = visual.cpu().numpy().astype(np.uint8)
                    print(visual.shape,visual.max())
                    visual = visual.transpose(1,2,0)
                    cv2.imwrite('visual/visual_' + str(i) + '.png',visual)
                '''
                features = []
                bs = 128
                f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10 = [],[],[],[],[],[],[],[],[],[]
                for i in range(0,batch.shape[0],bs):
                    if i+bs < batch.shape[0]:
                        cur_features = model.sample(batch[i:i+bs], clip_denoised=False)
                    #print(cur_features.shape)
                        #cur_features = cur_features.cpu().numpy()
                        #features.append(cur_features)
                        f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10 = restore_features(cur_features)
                    else:
                        cur_features = model.sample(batch[i:batch.shape[0]], clip_denoised=False)
                        f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10 = restore_features(cur_features)
                    #print(cur_features.shape)
                        #cur_features = cur_features.cpu().numpy()
                        #features.append(cur_features)
                f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10 = concat_features(f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10)
                print(f_1.shape,f_2.shape,f_3.shape,f_4.shape,f_5.shape,f_6.shape,f_7.shape,f_8.shape,f_9.shape,f_10.shape)
                asset_dict = {'f_1': f_1,'f_2': f_2,'f_3': f_3,'f_4': f_4,'f_5': f_5,'f_6': f_6,'f_7': f_7,'f_8': f_8,'f_9': f_9,'f_10':f_10, 'coords': coords}
                save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
                mode = 'a'
        
        return output_path
    @torch.no_grad()
    def sample_to_eval(self, net, sample_path):
        condition_path = make_dir(os.path.join(sample_path, f'condition'))
        gt_path = make_dir(os.path.join(sample_path, 'ground_truth'))
        result_path = make_dir(os.path.join(sample_path, str(self.config.model.BB.params.sample_step)))
     
        csv_path = 'preprocess_level1/process_list_autogen.csv'
        data_h5_dir = 'preprocess_level1/'
        data_slide_dir = 'DATA_DIRECTORY_CLAM/'
        feat_dir = 'camelyon_t=20_level1_features/'
  
        slide_ext = '.tif'
        
        os.makedirs(feat_dir, exist_ok=True)
        os.makedirs(os.path.join(feat_dir, 'pt_files'), exist_ok=True)
        os.makedirs(os.path.join(feat_dir, 'h5_files'), exist_ok=True)
        already_files = os.listdir(feat_dir+'h5_files/')
        bags_dataset = Dataset_All_Bags(csv_path)
        #pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01)
        batch_size = self.config.data.test.batch_size
        to_normal = self.config.data.dataset_config.to_normal
        sample_num = self.config.testing.sample_num
        
        total = len(bags_dataset)

        for bag_candidate_idx in range(total):
            slide_id = bags_dataset[bag_candidate_idx].split(slide_ext)[0]
            if slide_id+'.h5' not in already_files:
                bag_name = slide_id+'.h5'
                h5_file_path = os.path.join(data_h5_dir, 'patches', bag_name)
                slide_file_path = os.path.join(data_slide_dir, slide_id+slide_ext)
                print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
                print(slide_id)

                output_path = os.path.join(feat_dir, 'h5_files', bag_name)
                time_start = time.time()
                wsi = openslide.open_slide(slide_file_path)
                output_file_path = self.compute_w_loader(h5_file_path, output_path, wsi, 
                model = net, batch_size = 512, verbose = 1, print_every = 20, 
                custom_downsample=1, target_patch_size=-1)
                time_elapsed = time.time() - time_start
                print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
                file = h5py.File(output_file_path, "r")
                f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10 = file['f_1'][:],file['f_2'][:],file['f_3'][:],file['f_4'][:],file['f_5'][:],file['f_6'][:],file['f_7'][:],file['f_8'][:],file['f_9'][:],file['f_10'][:]
                f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8,f_9,f_10 = torch.from_numpy(f_1),torch.from_numpy(f_2),torch.from_numpy(f_3),torch.from_numpy(f_4),torch.from_numpy(f_5),torch.from_numpy(f_6),torch.from_numpy(f_7),torch.from_numpy(f_8),torch.from_numpy(f_9),torch.from_numpy(f_10)
                bag_base, _ = os.path.splitext(bag_name)
                torch.save(f_1, os.path.join(feat_dir, 'pt_files', bag_base+'_f1.pt'))
                torch.save(f_2, os.path.join(feat_dir, 'pt_files', bag_base+'_f2.pt'))
                torch.save(f_3, os.path.join(feat_dir, 'pt_files', bag_base+'_f3.pt'))
                torch.save(f_4, os.path.join(feat_dir, 'pt_files', bag_base+'_f4.pt'))
                torch.save(f_5, os.path.join(feat_dir, 'pt_files', bag_base+'_f5.pt'))
                torch.save(f_6, os.path.join(feat_dir, 'pt_files', bag_base+'_f6.pt'))
                torch.save(f_7, os.path.join(feat_dir, 'pt_files', bag_base+'_f7.pt'))
                torch.save(f_8, os.path.join(feat_dir, 'pt_files', bag_base+'_f8.pt'))
                torch.save(f_9, os.path.join(feat_dir, 'pt_files', bag_base+'_f9.pt'))
                torch.save(f_10, os.path.join(feat_dir, 'pt_files', bag_base+'_f10.pt'))
            
            
