 

import os
import glob
import torch
import shutil
import numpy as np
from torch import nn
from utils import common
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from collections import OrderedDict
from torch.utils.data import DataLoader
from dataset import MultipleModel, MultipleModelLink
from model_utils import (
    StateConditionMLPQueryModel,
    KinematicFeatToLinkModel,
    KinematicScratchModel
)

def rename_ckpt_for_multi_models(ckpt):
    renamed_state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        if k.split('.')[0] == 'model':
            name = k.replace('model.', '')
            renamed_state_dict[name] = v
    return renamed_state_dict

class VisModelingModel(pl.LightningModule):

    def __init__(self,
                 lr: float=5e-5,
                 seed: int=1,
                 dof: int=5,
                 if_cuda: bool=True,
                 if_test: bool=False,
                 gamma: float=0.5,
                 log_dir: str='logs',
                 train_batch: int=1400,
                 val_batch: int=1400,
                 test_batch: int=1400,
                 num_workers: int=8,
                 model_name: str='reconstruction',
                 data_filepath: str='data',
                 loss_type: str='siren_loss',
                 coord_system: str='cartesian',
                 lr_schedule: list=[100000]) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = {'num_workers': self.hparams.num_workers, 'pin_memory': True} if self.hparams.if_cuda else {}

        self.__build_model()

    def __build_model(self):
        # model
        if self.hparams.model_name == 'state-condition':
            self.model = StateConditionMLPQueryModel(in_channels=int(3+self.hparams.dof), out_channels=1, hidden_features=256)
        if self.hparams.model_name == 'state-condition-kinematic':
            self.model = KinematicFeatToLinkModel(in_channels=128, out_channels=3, hidden_features=64)
            self.state_condition_model = StateConditionMLPQueryModel(in_channels=int(3+self.hparams.dof), out_channels=1, hidden_features=256)
        if self.hparams.model_name == 'state-condition-kinematic-scratch':
            self.model = KinematicScratchModel(in_channels=4, out_channels=3, hidden_features=128, hidden_hidden_features=64)
        # loss
        if self.hparams.loss_type == 'siren_sdf':
            self.loss_func = self.siren_sdf_loss
        if self.hparams.loss_type == 'siren_sdf_kinematic':
            self.loss_func = nn.L1Loss()
        if self.hparams.loss_type == 'siren_sdf_kinematic_scratch':
            self.loss_func = nn.L1Loss()
    
    def extract_kinematic_encoder_model(self, state_condition_model_checkpoint_filepath):
        # load the original state conditional model
        state_condition_model_checkpoint_filepath = glob.glob(os.path.join(state_condition_model_checkpoint_filepath, '*.ckpt'))[0]
        ckpt = torch.load(state_condition_model_checkpoint_filepath)
        ckpt = rename_ckpt_for_multi_models(ckpt)
        self.state_condition_model.load_state_dict(ckpt)

        # freeze all the parameters in the state condition model
        for p in self.state_condition_model.parameters():
            p.requires_grad = False
        self.state_condition_model.eval()
    
    def l1_sdf_loss(self, model_output, gt):
        gt_sdf = gt['sdf'].reshape(-1, 1)
        pred_sdf = model_output['model_out']
        loss = F.l1_loss(pred_sdf, gt_sdf)
        return loss

    def siren_sdf_loss(self, model_output, gt):
        gt_sdf = gt['sdf']
        gt_normals = gt['normals']

        coords = model_output['model_in']
        pred_sdf = model_output['model_out']

        gt_sdf = gt_sdf.reshape(-1, 1)
        gt_normals = gt_normals.reshape(-1, 3)

        gradient = common.gradient(pred_sdf, coords)

        # wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
        sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
        inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
        normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                        torch.zeros_like(gradient[..., :1]))
        grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)

        loss_dict = {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
                     'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
                     'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
                     'grad_constraint': grad_constraint.mean() * 5e1}  # 1e1      # 5e1

        loss = loss_dict['sdf'] + loss_dict['inter'] + loss_dict['normal_constraint'] + loss_dict['grad_constraint']
        return loss
    
    def train_forward(self, data):
        if self.hparams.model_name == 'state-condition':
            data['coords'] = data['coords'].reshape(-1, 3)
            coords_org = data['coords'].clone().detach().requires_grad_(True)
            coords = coords_org
            states = data['states'].reshape(-1, self.hparams.dof)
            output = self.model(torch.cat((coords, states), dim=1))
            pred = {'model_in': coords_org, 'model_out': output}
            return pred

    def training_step(self, batch, batch_idx):
        data, target = batch
        if self.hparams.model_name == 'state-condition-kinematic':
            kinematic_feat = self.state_condition_model.state_encoder(data['states'])
            pred = self.model(kinematic_feat)
            train_loss = self.loss_func(pred, target['target_states'])
        elif self.hparams.model_name == 'state-condition-kinematic-scratch':
            pred = self.model(data['states'])
            train_loss = self.loss_func(pred, target['target_states'])
        else:
            pred = self.train_forward(data)
            train_loss = self.loss_func(pred, target)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    # def validation_step(self, batch, batch_idx):
    #     if self.hparams.model_name == 'state-condition-kinematic':
    #         data, target = batch
    #         kinematic_feat = self.state_condition_model.state_encoder(data['states'])
    #         pred = self.model(kinematic_feat)
    #         val_loss = self.loss_func(pred, target['target_states'])
    #         self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #         return val_loss
    #     elif self.hparams.model_name == 'state-condition-kinematic-scratch':
    #         data, target = batch
    #         pred = self.model(data['states'])
    #         val_loss = self.loss_func(pred, target['target_states'])
    #         self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #         return val_loss
    #     else:
    #         pass
        
    def test_step(self, batch, batch_idx):
        if self.hparams.model_name == 'state-condition-kinematic':
            data, target = batch
            kinematic_feat = self.state_condition_model.state_encoder(data['states'])
            pred = self.model(kinematic_feat)
            test_loss = self.loss_func(pred, target['target_states'])
            self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return test_loss
        elif self.hparams.model_name == 'state-condition-kinematic-scratch':
            data, target = batch
            pred = self.model(data['states'])
            test_loss = self.loss_func(pred, target['target_states'])
            self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            pass
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_schedule, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def setup(self, stage=None):
        if self.hparams.model_name == 'state-condition':
            if stage == 'fit':
                if self.hparams.loss_type == 'siren_sdf':
                    self.train_dataset = MultipleModel(flag='train',
                                                       seed=self.hparams.seed,
                                                       pointcloud_folder=self.hparams.data_filepath,
                                                       on_surface_points=self.hparams.train_batch)
            if stage == 'test':
                if self.hparams.loss_type == 'siren_sdf':
                    self.test_dataset = MultipleModel(flag='test',
                                                      seed=self.hparams.seed,
                                                      pointcloud_folder=self.hparams.data_filepath,
                                                      on_surface_points=self.hparams.test_batch)
        
        if self.hparams.model_name == 'state-condition-kinematic' or self.hparams.model_name == 'state-condition-kinematic-scratch':
            if stage == 'fit':
                self.train_dataset = MultipleModelLink(flag='train',
                                                       seed=self.hparams.seed,
                                                       pointcloud_folder=self.hparams.data_filepath)
                
                # self.val_dataset = MultipleModelLink(flag='val',
                #                                      seed=self.hparams.seed,
                #                                      pointcloud_folder=self.hparams.data_filepath)
            
            if stage == 'test':
                self.test_dataset = MultipleModelLink(flag='test',
                                                      seed=self.hparams.seed,
                                                      pointcloud_folder=self.hparams.data_filepath)


    def train_dataloader(self):
        if self.hparams.model_name == 'state-condition-kinematic' or self.hparams.model_name == 'state-condition-kinematic-scratch':
            train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                       batch_size=self.hparams.train_batch,
                                                       shuffle=True,
                                                       **self.kwargs)
        else:
            train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                       batch_size=32,
                                                       shuffle=True,
                                                       **self.kwargs)
        return train_loader

    # def val_dataloader(self):
    #     if self.hparams.model_name == 'state-condition-kinematic' or self.hparams.model_name == 'state-condition-kinematic-scratch':
    #         val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
    #                                                 batch_size=self.hparams.val_batch,
    #                                                 shuffle=False,
    #                                                 **self.kwargs)
    #     return val_loader

    def test_dataloader(self):
        if self.hparams.model_name == 'state-condition-kinematic' or self.hparams.model_name == 'state-condition-kinematic-scratch':
            test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                      batch_size=self.hparams.test_batch,
                                                      shuffle=False,
                                                      **self.kwargs)
        else:
            test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                      batch_size=32,
                                                      shuffle=False,
                                                      **self.kwargs)
        return test_loader