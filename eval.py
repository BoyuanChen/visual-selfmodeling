

import os
import sys
import glob
import yaml
import time
import json
import torch
import pprint
import shutil
import trimesh
import numpy as np
from tqdm import tqdm
from utils import common
from munch import munchify
from collections import OrderedDict
from models import VisModelingModel
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)

def create_state_condition_mesh():
    config_filepath = str(sys.argv[1])
    checkpoint_filepath = str(sys.argv[2])
    checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.model_name,
                        cfg.tag,
                        str(cfg.seed)])

    model = VisModelingModel(lr=cfg.lr,
                             seed=cfg.seed,
                             dof=cfg.dof,
                             if_cuda=cfg.if_cuda,
                             if_test=True,
                             gamma=cfg.gamma,
                             log_dir=log_dir,
                             train_batch=cfg.train_batch,
                             val_batch=cfg.val_batch,
                             test_batch=cfg.test_batch,
                             num_workers=cfg.num_workers,
                             model_name=cfg.model_name,
                             data_filepath=cfg.data_filepath,
                             loss_type=cfg.loss_type,
                             coord_system=cfg.coord_system,
                             lr_schedule=cfg.lr_schedule)

    ckpt = torch.load(checkpoint_filepath)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to('cuda')
    model.eval()
    model.freeze()
    
    # get test file ids
    with open(os.path.join('../assets', 'datainfo', f'multiple_models_data_split_dict_{cfg.seed}.json'), 'r') as file:
        seq_dict = json.load(file)
    id_lst = seq_dict['test']

    # get robot states
    robot_state_filepath = os.path.join(cfg.data_filepath, 'robot_state.json')
    with open(robot_state_filepath, 'r') as file:
        robot_state_dict = json.load(file)

    ply_save_folder = os.path.join(log_dir, 'predictions')
    common.mkdir(ply_save_folder)
    for idx in tqdm(id_lst):
        # get testing robot states
        sel_robot_state = np.array((robot_state_dict[str(idx)][0][0],
                                    robot_state_dict[str(idx)][1][0],
                                    robot_state_dict[str(idx)][2][0],
                                    robot_state_dict[str(idx)][3][0])).reshape(1, -1)
        sel_robot_state = sel_robot_state / np.pi

        N=256
        max_batch=64 ** 3
        ply_filename = os.path.join(ply_save_folder, str(idx))
        start = time.time()
        
        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)

        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 3, 4)

        # transform first 3 columns to be the x, y, z index
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long() / N) % N
        samples[:, 0] = ((overall_index.long() / N) / N) % N

        # transform first 3 columns to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

        num_samples = N ** 3
        samples.requires_grad = False
        head = 0

        while head < num_samples:
            print(head)
            sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
            final_robot_states = np.tile(sel_robot_state, (sample_subset.shape[0], 1))
            final_robot_states = torch.from_numpy(final_robot_states).float().cuda()
            sample_subset = torch.cat((sample_subset, final_robot_states), dim=1)

            samples[head : min(head + max_batch, num_samples), 3] = (model.model(sample_subset).squeeze().detach().cpu())
            head += max_batch

        sdf_values = samples[:, 3]
        sdf_values = sdf_values.reshape(N, N, N)

        end = time.time()
        print("sampling takes: %f" % (end - start))

        common.convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            ply_filename + ".ply",
            offset=None,
            scale=None,
        )

# render predictions as angle smooth movements as animation
def create_state_condition_mesh_render():
    config_filepath = str(sys.argv[1])
    checkpoint_filepath = str(sys.argv[2])
    checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.model_name,
                        cfg.tag,
                        str(cfg.seed)])

    model = VisModelingModel(lr=cfg.lr,
                             seed=cfg.seed,
                             dof=cfg.dof,
                             if_cuda=cfg.if_cuda,
                             if_test=True,
                             gamma=cfg.gamma,
                             log_dir=log_dir,
                             train_batch=cfg.train_batch,
                             val_batch=cfg.val_batch,
                             test_batch=cfg.test_batch,
                             num_workers=cfg.num_workers,
                             model_name=cfg.model_name,
                             data_filepath=cfg.data_filepath,
                             loss_type=cfg.loss_type,
                             coord_system=cfg.coord_system,
                             lr_schedule=cfg.lr_schedule)

    ckpt = torch.load(checkpoint_filepath)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to('cuda')
    model.eval()
    model.freeze()

    # ############ multiple joint movements as a sequence ############
    animation_flag = 'render'
    base_folder = '/data/bo/saved_meshes_animation'
    force_connectivity = True

    folder_seqs = os.listdir(base_folder)
    robot_state_seqs = {}
    for p_seq_folder in folder_seqs:
        seq_index = p_seq_folder.split('_')[1]
        folder = os.path.join(base_folder, p_seq_folder)
        robot_state_filepath = os.path.join(base_folder, p_seq_folder, 'robot_state.json')
        with open(robot_state_filepath, 'r') as file:
            robot_state_dict = json.load(file)
        temp_robot_state_seqs = []
        num_keys = len(list(robot_state_dict.keys()))
        for i in range(num_keys):
            state_vector = [robot_state_dict[str(i)][0][0],
                            robot_state_dict[str(i)][1][0],
                            robot_state_dict[str(i)][2][0],
                            robot_state_dict[str(i)][3][0]]
            temp_robot_state_seqs.append(state_vector)
        robot_state_seqs[seq_index] = temp_robot_state_seqs

    for seq_index in list(robot_state_seqs.keys()):
        ply_save_folder = os.path.join(log_dir, f'prediction_{animation_flag}', f'sequence_{seq_index}')
        common.mkdir(ply_save_folder)

        num_renderings = len(robot_state_seqs[seq_index])
        for idx in tqdm(range(num_renderings)):
            sel_robot_state = np.array(robot_state_seqs[seq_index][idx]) / np.pi

            N=256
            max_batch=64 ** 3
            ply_filename = os.path.join(ply_save_folder, f'mesh_{idx}.ply')
            start = time.time()
            
            # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
            voxel_origin = [-1, -1, -1]
            voxel_size = 2.0 / (N - 1)

            overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
            samples = torch.zeros(N ** 3, 4)

            # transform first 3 columns to be the x, y, z index
            samples[:, 2] = overall_index % N
            samples[:, 1] = (overall_index.long() / N) % N
            samples[:, 0] = ((overall_index.long() / N) / N) % N

            # transform first 3 columns to be the x, y, z coordinate
            samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
            samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
            samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

            num_samples = N ** 3
            samples.requires_grad = False
            head = 0

            while head < num_samples:
                print(head)
                sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
                final_robot_states = np.tile(sel_robot_state, (sample_subset.shape[0], 1))
                final_robot_states = torch.from_numpy(final_robot_states).float().cuda()
                sample_subset = torch.cat((sample_subset, final_robot_states), dim=1)

                samples[head : min(head + max_batch, num_samples), 3] = (model.model(sample_subset).squeeze().detach().cpu())
                head += max_batch

            sdf_values = samples[:, 3]
            sdf_values = sdf_values.reshape(N, N, N)

            end = time.time()
            print("sampling takes: %f" % (end - start))

            common.convert_sdf_samples_to_ply(
                sdf_values.data.cpu(),
                voxel_origin,
                voxel_size,
                ply_filename,
                offset=None,
                scale=None,
            )

            # denormalize
            pred_mesh = trimesh.load(ply_filename)
            pred_mesh.vertices[:, 0] = pred_mesh.vertices[:, 0] * 0.45
            pred_mesh.vertices[:, 1] = pred_mesh.vertices[:, 1] * 0.45
            pred_mesh.vertices[:, 2] = ((pred_mesh.vertices[:, 2] * 0.5) + 0.5) * (0.51 + 0.13)

            if force_connectivity:
                # remove disconnected components from the predicted mesh
                pred_mesh = pred_mesh.as_open3d
                triangle_clusters, cluster_n_triangles, cluster_area = (pred_mesh.cluster_connected_triangles())
                triangle_clusters = np.asarray(triangle_clusters)
                cluster_n_triangles = np.asarray(cluster_n_triangles)
                cluster_area = np.asarray(cluster_area)
                triangles_to_remove = cluster_n_triangles[triangle_clusters] < 1000

                largest_cluster_idx = cluster_n_triangles.argmax()
                triangles_to_remove = triangle_clusters != largest_cluster_idx
                pred_mesh.remove_triangles_by_mask(triangles_to_remove)
                # o3d.visualization.draw_geometries([mesh_1]) # for debugging visualization only

                # get trimesh from o3d mesh
                pred_mesh = trimesh.Trimesh(np.asarray(pred_mesh.vertices),
                                            np.asarray(pred_mesh.triangles),
                                            vertex_normals=np.asarray(pred_mesh.vertex_normals))
            
            os.remove(ply_filename)
            pred_mesh.export(ply_filename)

def evaluate_kinematic():
    config_filepath = str(sys.argv[1])
    checkpoint_filepath = str(sys.argv[2])
    checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.model_name,
                        cfg.tag,
                        str(cfg.seed)])

    model = VisModelingModel(lr=cfg.lr,
                             seed=cfg.seed,
                             dof=cfg.dof,
                             if_cuda=cfg.if_cuda,
                             if_test=True,
                             gamma=cfg.gamma,
                             log_dir=log_dir,
                             train_batch=cfg.train_batch,
                             val_batch=cfg.val_batch,
                             test_batch=cfg.test_batch,
                             num_workers=cfg.num_workers,
                             model_name=cfg.model_name,
                             data_filepath=cfg.data_filepath,
                             loss_type=cfg.loss_type,
                             coord_system=cfg.coord_system,
                             lr_schedule=cfg.lr_schedule)

    ckpt = torch.load(checkpoint_filepath)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to('cuda')
    model.eval()
    model.freeze()
    model.extract_kinematic_encoder_model(sys.argv[4])

    trainer = Trainer(gpus=cfg.num_gpus,
                      max_epochs=cfg.epochs,
                      deterministic=True,
                      amp_backend='native',
                      default_root_dir=log_dir,
                      val_check_interval=1.0)
    trainer.test(model)

def evaluate_kinematic_scratch():
    config_filepath = str(sys.argv[1])
    checkpoint_filepath = str(sys.argv[2])
    checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.model_name,
                        cfg.tag,
                        str(cfg.seed)])

    model = VisModelingModel(lr=cfg.lr,
                             seed=cfg.seed,
                             dof=cfg.dof,
                             if_cuda=cfg.if_cuda,
                             if_test=True,
                             gamma=cfg.gamma,
                             log_dir=log_dir,
                             train_batch=cfg.train_batch,
                             val_batch=cfg.val_batch,
                             test_batch=cfg.test_batch,
                             num_workers=cfg.num_workers,
                             model_name=cfg.model_name,
                             data_filepath=cfg.data_filepath,
                             loss_type=cfg.loss_type,
                             coord_system=cfg.coord_system,
                             lr_schedule=cfg.lr_schedule)

    ckpt = torch.load(checkpoint_filepath)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to('cuda')
    model.eval()
    model.freeze()

    trainer = Trainer(gpus=cfg.num_gpus,
                      max_epochs=cfg.epochs,
                      deterministic=True,
                      amp_backend='native',
                      default_root_dir=log_dir,
                      val_check_interval=1.0)
    trainer.test(model)

if __name__ == '__main__':
    if sys.argv[3] == 'eval-state-condition':
        create_state_condition_mesh()
    if sys.argv[3] == 'eval-state-condition-animation':
        create_state_condition_mesh_render()
    if sys.argv[3] == 'eval-kinematic':
        evaluate_kinematic()
    if sys.argv[3] == 'eval-kinematic-scratch':
        evaluate_kinematic_scratch()