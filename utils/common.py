
import os
import glob
import time
import json
import torch
import random
import shutil
import logging
import plyfile
import trimesh
import numpy as np
import pyvista as pv
from tqdm import tqdm
import skimage.measure
from torch.autograd import grad
from mesh_to_sdf import sample_sdf_near_surface, get_surface_point_cloud, scale_to_unit_sphere

def set_visible(client, visual_data, visible=True):
    if visual_data is None:
        return
    for body_id, link_colors in visual_data.items():
        for link_ind, link_color in link_colors.items():
            client.changeVisualShape(
                body_id, link_ind,
                rgbaColor=link_color if visible else (0, 0, 0, 0)
            )

def get_body_colors(client, body_id):
    link_colors = dict()
    for visual_info in client.getVisualShapeData(body_id):
        link_colors[visual_info[1]] = visual_info[7]
    return link_colors

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def gradient(y, x, grad_outputs=None):
    """reference: https://github.com/vsitzmann/siren"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    except:
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )




def create_mesh(
    decoder, filename, N=1600, max_batch=64 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        print(head)
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            decoder(sample_subset)
            .squeeze()#.squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )

def create_random_seed_split(seed=1, ratio=0.9):
    seq_dict = {}
    total_num = 10000

    ids = list(range(total_num))
    random.shuffle(ids)

    # test
    start = int(total_num * ratio)
    seq_dict['test'] = ids[start:]
    
    # train
    seq_dict['train'] = ids[:int(total_num*ratio)]

    with open(os.path.join(f'../assets/datainfo/multiple_models_data_split_dict_{seed}.json'), 'w') as file:
        json.dump(seq_dict, file, indent=4)

def convert_ply_to_xyzn(folder='./saved_meshes'):
    import open3d as o3d
    all_ply_files = glob.glob(os.path.join(folder, 'mesh_*.ply'))
    for p_file in tqdm(all_ply_files):
        mesh = o3d.io.read_triangle_mesh(p_file)
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.colors = mesh.vertex_colors
        pcd.normals = mesh.vertex_normals
        o3d.io.write_point_cloud(p_file.replace('ply', 'xyzn'), pcd)

def convert_ply_to_sdf_old(folder='./saved_meshes'):
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    all_ply_files = glob.glob(os.path.join(folder, 'mesh_*.ply'))
    for p_file in tqdm(all_ply_files):
        mesh = trimesh.load(p_file)
        mesh = scale_to_unit_sphere(mesh)
        surface_point_cloud = get_surface_point_cloud(mesh)
        points, sdf = surface_point_cloud.sample_sdf_near_surface(number_of_points=250000)
        np.savez(p_file.replace('ply', 'npz'), points=points, sdf=sdf)

def convert_ply_to_sdf(p_file):
    mesh = trimesh.load(p_file)
    points, sdf = sample_sdf_near_surface(mesh, 250000)
    translation, scale = compute_unit_sphere_transform(mesh)
    points = (points / scale) - translation
    sdf /= scale
    np.savez(p_file.replace('ply', 'npz'), points=points, sdf=sdf)

def convert_ply_to_sdf_parallel(folder='./saved_meshes'):
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    from multiprocessing import Pool
    pool = Pool(8)

    all_ply_files = glob.glob(os.path.join(folder, 'mesh_*.ply'))
    progress = tqdm(total=len(all_ply_files))
    def on_complete(*_):
        progress.update()

    for p_file in all_ply_files:
        pool.apply_async(convert_ply_to_sdf, args=(p_file,), callback=on_complete)
    pool.close()
    pool.join()

def compute_unit_sphere_transform(mesh):
    """
    returns translation and scale, which is applied to meshes before computing their SDF cloud
    """
    # the transformation applied by mesh_to_sdf.scale_to_unit_sphere(mesh)
    translation = -mesh.bounding_box.centroid
    scale = 1 / np.max(np.linalg.norm(mesh.vertices + translation, axis=1))
    return translation, scale

# cmake -GNinja -DVTK_BUILD_TESTING=OFF -DVTK_WHEEL_BUILD=ON -DVTK_PYTHON_VERSION=3 -DVTK_WRAP_PYTHON=ON -DVTK_OPENGL_HAS_EGL=True -DVTK_USE_X=False -DPython3_EXECUTABLE=$PYBIN ../
def render_screenshot_for_multiple_conditional_angles(folder, save_to_folder):
    mkdir(save_to_folder)
    pv.set_plot_theme("document")

    filenames = os.listdir(folder)
    angle_lst = []
    for p_filename in filenames:
        angle_lst.append(float(p_filename.replace('.ply', '')))
    angle_lst = sorted(angle_lst)

    for p_angle in tqdm(angle_lst):
        filename = str(p_angle) + '.ply'
        filepath = os.path.join(folder, filename)
        mesh = pv.read(filepath)

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh, color='#454545')
        plotter.add_text(str(p_angle), name='angle')
        plotter.show(screenshot=os.path.join(save_to_folder, f'{p_angle}.png'))
        plotter.close()

def render_saved_screenshot_to_video(folder='angles_rendered_angle_0_pi_pi'):
    save_to_folder = folder+'_renamed'
    mkdir(save_to_folder)

    filenames = os.listdir(folder)
    angle_lst = []
    for p_filename in filenames:
        angle_lst.append(float(p_filename.replace('.png', '')))
    angle_lst = sorted(angle_lst)

    for idx in range(len(angle_lst)):
        os.rename(os.path.join(folder, f'{angle_lst[idx]}.png'), os.path.join(save_to_folder, f'{idx}.png'))

    os.system(f'ffmpeg -r 1 -i {save_to_folder}/%01d.png -vcodec mpeg4 -y {folder}.mp4')