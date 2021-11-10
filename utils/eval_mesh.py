
import random
import common
import trimesh
import logging
import numpy as np
from numpy.lib.financial import ipmt
from scipy import spatial
from libkdtree import KDTree
from libmesh import check_mesh_contains

logger = logging.getLogger(__name__)


# adapted from https://github.com/autonomousvision/occupancy_networks/
class MeshEvaluator(object):
    ''' 
    Args:
        n_points (int): number of points to be used for evaluation. Default: 100k
    '''

    def __init__(self, n_points=100000):
        self.n_points = n_points

    def eval_mesh(self, mesh, mesh_tgt):
        ''' Evaluates a mesh.
        Args:
            mesh (trimesh): mesh which should be evaluated
            mesh_tgt (trimesh): ground-truth mesh
        '''
        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            pointcloud, idx = mesh.sample(self.n_points, return_index=True)
            pointcloud = pointcloud.astype(np.float32)
            normals = mesh.face_normals[idx]
        else:
            pointcloud = np.empty((0, 3))
            normals = np.empty((0, 3))

        pointcloud_tgt, idx = mesh_tgt.sample(self.n_points, return_index=True)
        pointcloud_tgt = pointcloud_tgt.astype(np.float32)
        normals_tgt = mesh_tgt.face_normals[idx]

        out_dict = self.eval_pointcloud(pointcloud, pointcloud_tgt, normals, normals_tgt)

        return out_dict

    def eval_pointcloud(self, pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None):
        ''' Evaluates a point cloud.
        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        '''
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            import IPython
            IPython.embed()
            assert False

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from the predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are the points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamferL2,
            'chamfer-L1': chamferL1,
        }

        return out_dict

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product

import os
import sys
import glob
import yaml
import time
import json
import pprint
import shutil
import numpy as np
import open3d as o3d
from tqdm import tqdm
from munch import munchify


def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def eval_mesh_main():
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)

    # get the folder to save the eval out_dict results
    log_dir = '_'.join([cfg.log_dir,
                        cfg.model_name,
                        cfg.tag,
                        str(cfg.seed)])
    prediction_folder = os.path.join('../scripts', log_dir, 'predictions')
    denormalized_prediction_folder = os.path.join('../scripts', log_dir, 'prediction_denormalized')
    common.mkdir(denormalized_prediction_folder)

    # get test file ids
    with open(os.path.join('../assets', 'datainfo', f'multiple_models_data_split_dict_{cfg.seed}.json'), 'r') as file:
        seq_dict = json.load(file)
    id_lst = seq_dict['test']

    start = None
    eval_dict = {}
    mesh_evaluator = MeshEvaluator(n_points=100000)
    for idx in tqdm(id_lst):
        pred_mesh_filepath = os.path.join(prediction_folder, f'{idx}.ply')
        gt_mesh_filepath = os.path.join(cfg.data_filepath, f'mesh_{idx}.ply')

        pred_mesh = trimesh.load(pred_mesh_filepath)
        gt_mesh = trimesh.load(gt_mesh_filepath)

        # de-normalize the predicted mesh
        pred_mesh.vertices[:, 0] = pred_mesh.vertices[:, 0] * 0.45
        pred_mesh.vertices[:, 1] = pred_mesh.vertices[:, 1] * 0.45
        pred_mesh.vertices[:, 2] = ((pred_mesh.vertices[:, 2] * 0.5) + 0.5) * (0.51 + 0.13)
        pred_mesh = pred_mesh.as_open3d

        # remove disconnected components from the predicted mesh
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
        
        # evaluate the prediction
        per_eval_dict = mesh_evaluator.eval_mesh(pred_mesh, gt_mesh)
        # update the eval_dict
        if start is None:
            for k, v in per_eval_dict.items():
                eval_dict[k] = [v]
            start = True
        else:
            for k, v in per_eval_dict.items():
                eval_dict[k].append(v)
        print(per_eval_dict['chamfer-L1'])

        # save the denormalized new mesh
        pred_mesh.export(os.path.join(denormalized_prediction_folder, f'{idx}.ply'))

    for p_key in list(eval_dict.keys()):
        total_num = len(eval_dict[p_key])
        for idx in range(total_num):
            eval_dict[p_key][idx] = np.float64(eval_dict[p_key][idx])

    with open(os.path.join(denormalized_prediction_folder, 'eval_dict.json'), 'w') as file:
        json.dump(eval_dict, file, indent=4)

def eval_mesh_nearest_neighbor_main():
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)

    # get the folder to save the eval out_dict results
    log_dir = '_'.join([cfg.log_dir,
                        cfg.model_name,
                        cfg.tag,
                        str(cfg.seed)])
    prediction_folder = os.path.join('../scripts', log_dir, 'predictions')

    # get test file ids
    with open(os.path.join('../assets', 'datainfo', f'multiple_models_data_split_dict_{cfg.seed}.json'), 'r') as file:
        seq_dict = json.load(file)
    id_lst = seq_dict['test']
    train_id_lst = seq_dict['train']

    # get the robot state dict
    robot_state_filepath = os.path.join(cfg.data_filepath, 'robot_state.json')
    with open(robot_state_filepath, 'r') as file:
        robot_state_dict = json.load(file)
    
    train_robot_state_value_lst = []
    train_robot_state_id_lst = []

    for idx in train_id_lst:
        robot_state = robot_state_dict[str(idx)]
        train_robot_state_value_lst.append(np.array([robot_state[0][0], robot_state[1][0], robot_state[2][0], robot_state[3][0], robot_state[4][0]]))
        train_robot_state_id_lst.append(idx)
    train_tree = spatial.KDTree(train_robot_state_value_lst)
    
    start = None
    eval_dict = {}
    mesh_evaluator = MeshEvaluator(n_points=100000)
    for idx in tqdm(id_lst):
        # get the robot state for the current testing data
        robot_state = robot_state_dict[str(idx)]
        test_robot_state = np.array([robot_state[0][0], robot_state[1][0], robot_state[2][0], robot_state[3][0], robot_state[4][0]]).reshape(1, -1)
        # get the nearest data point from the training data
        dist, nn_index = train_tree.query(test_robot_state)
        nn_train_index = train_robot_state_id_lst[nn_index.item()]

        # compare
        pred_mesh_filepath = os.path.join(cfg.data_filepath, f'mesh_{nn_train_index}.ply')
        gt_mesh_filepath = os.path.join(cfg.data_filepath, f'mesh_{idx}.ply')

        pred_mesh = o3d.io.read_triangle_mesh(pred_mesh_filepath)
        gt_mesh = trimesh.load(gt_mesh_filepath)

        # remove disconnected components from the predicted mesh
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
        
        # evaluate the prediction
        per_eval_dict = mesh_evaluator.eval_mesh(pred_mesh, gt_mesh)
        # update the eval_dict
        if start is None:
            for k, v in per_eval_dict.items():
                eval_dict[k] = [v]
            start = True
        else:
            for k, v in per_eval_dict.items():
                eval_dict[k].append(v)
        print(per_eval_dict['chamfer-L1'])

    for p_key in list(eval_dict.keys()):
        total_num = len(eval_dict[p_key])
        for idx in range(total_num):
            eval_dict[p_key][idx] = np.float64(eval_dict[p_key][idx])

    with open(os.path.join(prediction_folder, 'eval_nearest_neighbor_dict.json'), 'w') as file:
        json.dump(eval_dict, file, indent=4)


def eval_mesh_random_main():
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)

    # get the folder to save the eval out_dict results
    log_dir = '_'.join([cfg.log_dir,
                        cfg.model_name,
                        cfg.tag,
                        str(cfg.seed)])
    prediction_folder = os.path.join('../scripts', log_dir, 'predictions')

    # get test file ids
    with open(os.path.join('../assets', 'datainfo', f'multiple_models_data_split_dict_{cfg.seed}.json'), 'r') as file:
        seq_dict = json.load(file)
    id_lst = seq_dict['test']
    train_id_lst = seq_dict['train']
    
    start = None
    eval_dict = {}
    mesh_evaluator = MeshEvaluator(n_points=100000)
    for idx in tqdm(id_lst):
        # get a random index from the training set
        random_train_idx = random.choice(train_id_lst)
        # compare
        pred_mesh_filepath = os.path.join(cfg.data_filepath, f'mesh_{random_train_idx}.ply')
        gt_mesh_filepath = os.path.join(cfg.data_filepath, f'mesh_{idx}.ply')

        pred_mesh = o3d.io.read_triangle_mesh(pred_mesh_filepath)
        gt_mesh = trimesh.load(gt_mesh_filepath)

        # remove disconnected components from the predicted mesh
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
        
        # evaluate the prediction
        per_eval_dict = mesh_evaluator.eval_mesh(pred_mesh, gt_mesh)
        # update the eval_dict
        if start is None:
            for k, v in per_eval_dict.items():
                eval_dict[k] = [v]
            start = True
        else:
            for k, v in per_eval_dict.items():
                eval_dict[k].append(v)
        print(per_eval_dict['chamfer-L1'])

    for p_key in list(eval_dict.keys()):
        total_num = len(eval_dict[p_key])
        for idx in range(total_num):
            eval_dict[p_key][idx] = np.float64(eval_dict[p_key][idx])

    with open(os.path.join(prediction_folder, 'eval_random_dict.json'), 'w') as file:
        json.dump(eval_dict, file, indent=4)

if __name__ == '__main__':
    algorithm = sys.argv[2]
    if algorithm == 'model':
        eval_mesh_main()
    if algorithm == 'nearest-neighbor':
        eval_mesh_nearest_neighbor_main()
    if algorithm == 'random':
        eval_mesh_random_main()