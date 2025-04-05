import os
import sys
import torch
from scipy.spatial import cKDTree
from CryFold.utils.mrc_tools import load_map,make_model_grid
from CryFold.utils.network_tools import map_segmentation,map_reconstruction
from CryFold.utils.save_pdb_utils import points_to_pdb
from CryFold.utils.torch_utlis import get_batch_slices
from CryFold.Unet.Unet import SimpleUnet
import numpy as np
import tqdm

def get_lattice_meshgrid_np(shape_size, no_shift=False):
    linspace = [np.linspace(
        0.5 if not no_shift else 0,
        shape - (0.5 if not no_shift else 1),
        shape,
    ) for shape in shape_size]
    mesh = np.stack(
        np.meshgrid(linspace[0], linspace[1], linspace[2], indexing="ij"),
        axis=-1,
    )
    return mesh
def grid_to_points(grid, threshold, neighbour_distance_threshold):
    """
    MIT License

    Copyright (c) 2022 Kiarash Jamali

    This function comes from: https://github.com/3dem/model-angelo/blob/main/model_angelo/c_alpha/inference.py

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
    """
    lattice = np.flip(get_lattice_meshgrid_np(grid.shape, no_shift=False), -1)

    output_points_before_pruning = np.copy(lattice[grid > threshold, :].reshape(-1, 3))

    points = lattice[grid > threshold, :].reshape(-1, 3)
    probs = grid[grid > threshold]
    # sorted_indices = np.argsort(probs)[::-1]
    # probs = probs[sorted_indices]
    # points = points[sorted_indices]
    for _ in range(3):
        kdtree = cKDTree(np.copy(points))
        n = 0
        new_points = np.copy(points)
        for p in points:
            neighbours = kdtree.query_ball_point(p,1.1)
            selection = list(neighbours)
            if len(neighbours) > 1 and np.sum(probs[selection]) > 0:
                keep_idx = np.argmax(probs[selection])
                prob_sum = np.sum(probs[selection])

                new_points[selection[keep_idx]] = (
                    np.sum(probs[selection][..., None] * points[selection], axis=0)
                    / prob_sum
                )
                probs[selection] = 0
                probs[selection[keep_idx]] = prob_sum

            n += 1

        points = new_points[probs > 0].reshape(-1, 3)
        probs = probs[probs > 0]

    kdtree = cKDTree(np.copy(points))
    for point_idx, point in enumerate(points):
        d, _ = kdtree.query(point, 2)
        if d[1] > neighbour_distance_threshold:
            points[point_idx] = np.nan

    points = points[~np.isnan(points).any(axis=-1)].reshape(-1, 3)

    output_points = points
    return output_points, output_points_before_pruning

def predict_slide(grid,model,stride: int = 100, windows_size: int = 129, batch_size: int = 1,device='cpu'):
    segmentation = map_segmentation(torch.tensor(grid, dtype=torch.float), stride=stride, windows_size=windows_size)
    segmentation = torch.stack(segmentation, dim=0)
    segmentation = segmentation[:, None]
    grid_batches = get_batch_slices(segmentation.shape[0], batch_size)
    with torch.no_grad():
        segmentation = segmentation.to(device)
        out_segmentation = torch.zeros(segmentation.shape, device=device)
        for grid_batch in grid_batches:
            out_segmentation[grid_batch] = torch.sigmoid(model(segmentation[grid_batch]))
        out_segmentation = out_segmentation[:, 0]
        out_segmentation = out_segmentation.detach().cpu().numpy()
    pred = map_reconstruction(out_segmentation, grid.shape, stride=stride, windows_size=windows_size)
    return pred

def infer(args):
    device = torch.device(args.device)
    os.makedirs(args.output_path, exist_ok=True)
    model_output_dir = os.path.join(args.output_path, "see_alpha_output_ca.cif")
    module = SimpleUnet()
    module.load_state_dict(torch.load(args.log_dir,map_location=torch.device('cpu')))
    module.to(device)
    module.eval()
    if args.map_path.endswith("map") or args.map_path.endswith("mrc"):
        grid_np, voxel_size, global_origin = load_map(args.map_path)
        if args.mask_path:
            assert args.mask_path.endswith("map") or args.mask_path.endswith("mrc")
            mask_np,b1,b2 = load_map(args.mask_path)
        else:
            mask_np = np.ones(grid_np.shape)
        grid_np = grid_np*mask_np
        grid_np, voxel_size, global_origin = make_model_grid(
            np.copy(grid_np), voxel_size, global_origin, target_voxel_size=1.5
        )
    else:
        raise RuntimeError(f"File {args.map_path} is not a cryo-em density map file format.")
    grid_np = (grid_np - np.mean(grid_np)) / np.std(grid_np)
    grid = (grid_np).astype(np.float32)
    batch_size = int(args.batch_size)
    windows_size = args.windows_size
    stride = args.stride
    shape = np.array(grid_np.shape[-3:])
    total_batch_num = np.prod(np.ceil(shape/stride))

    pbar = tqdm.tqdm(
        total=total_batch_num,
        file=sys.stdout,
        position=0,
        leave=True,
    )
    if np.all(shape>windows_size):
        segmentation = map_segmentation(torch.from_numpy(grid), stride=stride, windows_size=windows_size)
        segmentation = torch.stack(segmentation, dim=0)
        segmentation = segmentation[:, None]
        grid_batches = get_batch_slices(segmentation.shape[0], batch_size)
        with torch.no_grad():
            segmentation = segmentation.to(device)
            out_segmentation = torch.zeros(segmentation.shape, device=device)
            for grid_batch in grid_batches:
                out_segmentation[grid_batch] = torch.sigmoid(module(segmentation[grid_batch]))
                pbar.update(batch_size)
            out_segmentation = out_segmentation[:, 0]
            out_segmentation = out_segmentation.detach().cpu().numpy()
        pred = map_reconstruction(out_segmentation, grid.shape, stride=stride, windows_size=windows_size)
    else:
        with torch.no_grad():
            pred = torch.sigmoid(module(torch.from_numpy(grid)[None][None].to(device)))[0,0].detach().cpu().numpy()
            pbar.update(1)
    pbar.close()


    output_ca_points, output_ca_points_before_pruning = grid_to_points(
        pred,threshold=args.threshold,neighbour_distance_threshold=6/np.min(voxel_size)
    )

    # 将点集转换回密度图并保存
    try:
        import mrcfile
        
        # 将点集转换回密度图
        def points_to_grid(points, shape):
            """
            将点集转换为密度图
            """
            grid = np.zeros(shape, dtype=np.float32)
            for point in points:
                i, j, k = np.round(point).astype(int)
                if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
                    grid[i, j, k] = 1.0
            return grid
        
        # 保存处理前的点集密度图
        before_pruning_grid = points_to_grid(output_ca_points_before_pruning, pred.shape)
        before_pruning_path = os.path.join(args.output_path, "before_pruning.mrc")
        
        # 保存处理后的点集密度图
        after_pruning_grid = points_to_grid(output_ca_points, pred.shape)
        after_pruning_path = os.path.join(args.output_path, "after_pruning.mrc")
        
        # 保存原始预测密度图以便比较
        pred_path = os.path.join(args.output_path, "prediction.mrc")
        
        # 定义本地版本的save_dens_map函数
        def save_dens_map(save_map_path, new_dens, current_voxel_size, current_origin):
            """
            保存密度图为MRC文件，使用当前的体素大小和原点
            
            参数：
            save_map_path: 保存路径
            new_dens: 新的密度图数据（预测结果）
            current_voxel_size: 当前体素大小
            current_origin: 当前全局原点
            """
            data_new = np.float32(new_dens)
            
            # 创建新的MRC文件
            with mrcfile.new(save_map_path, overwrite=True) as mrc_new:
                # 设置数据
                mrc_new.set_data(data_new)
                
                # 设置体素大小
                vsize = mrc_new.voxel_size
                vsize.flags.writeable = True
                
                # 确保体素大小是float类型
                if isinstance(current_voxel_size, np.ndarray):
                    if current_voxel_size.size >= 3:
                        vsize.x = float(current_voxel_size[0])
                        vsize.y = float(current_voxel_size[1])
                        vsize.z = float(current_voxel_size[2])
                    elif current_voxel_size.size == 1:
                        vs_value = float(current_voxel_size.item())
                        vsize.x = vs_value
                        vsize.y = vs_value
                        vsize.z = vs_value
                else:
                    vsize.x = float(current_voxel_size)
                    vsize.y = float(current_voxel_size)
                    vsize.z = float(current_voxel_size)
                
                mrc_new.voxel_size = vsize
                
                # 设置原点
                if isinstance(current_origin, np.ndarray):
                    if current_origin.size >= 3:
                        mrc_new.header.origin.x = float(current_origin[0])
                        mrc_new.header.origin.y = float(current_origin[1])
                        mrc_new.header.origin.z = float(current_origin[2])
                    elif current_origin.size == 1:
                        orig_value = float(current_origin.item())
                        mrc_new.header.origin.x = orig_value
                        mrc_new.header.origin.y = orig_value
                        mrc_new.header.origin.z = orig_value
                else:
                    mrc_new.header.origin.x = float(current_origin)
                    mrc_new.header.origin.y = float(current_origin)
                    mrc_new.header.origin.z = float(current_origin)
                
                # 设置标准轴排列
                mrc_new.header.mapc = 1  # 按照标准约定: mapc=1, mapr=2, maps=3
                mrc_new.header.mapr = 2
                mrc_new.header.maps = 3
                
                # 更新头部统计信息
                mrc_new.update_header_stats()
                
        # 确保输出目录存在
        os.makedirs(args.output_path, exist_ok=True)
        
        # 保存所有密度图文件
        # 先保存预测密度图
        save_dens_map(pred_path, pred, voxel_size, global_origin)
        print(f"已保存原始预测密度图: {pred_path}")
        
        # 保存处理前的点集密度图
        save_dens_map(before_pruning_path, before_pruning_grid, voxel_size, global_origin)
        print(f"已保存处理前点集密度图: {before_pruning_path}")
        
        # 保存处理后的点集密度图
        save_dens_map(after_pruning_path, after_pruning_grid, voxel_size, global_origin)
        print(f"已保存处理后点集密度图: {after_pruning_path}")
        
        # 使用预测时的体素大小和原点保存另一个版本的密度图（如之前代码所示）
        output_mrc_path = os.path.join(args.output_path, "unet_prediction.mrc")
        save_dens_map(output_mrc_path, grid, voxel_size, global_origin)
        print(f"已保存UNet预测密度图: {output_mrc_path}")
    except Exception as e:
        print(f"保存密度图时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    points_to_pdb(
        os.path.join(args.output_path, "output_ca_points_before_pruning.cif"),
        output_ca_points_before_pruning * voxel_size[None] + global_origin[None],
    )
    output_file_path = os.path.join(args.output_path, "see_alpha_output_ca.cif")
    points_to_pdb(
        output_file_path,
        output_ca_points * voxel_size[None] + global_origin[None],
    )

    return model_output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map-path", "--v", required=True, help="input cryo-em density map"
    )
    parser.add_argument(
        "--mask-path", "--m", required=True, help="input cryo-em mask map"
    )
    parser.add_argument(
        "--output-path",
        "--o",
        required=True,
        help="The C-alpha atoms ouput path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="compute device, pick one of {cpu, cuda:number}. "
             "Default set to use cpu.",
        help="The device to carry computations on",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for inference"
    )
    parser.add_argument(
        "--stride", type=int, default=100, help="The stride for inference"
    )
    parser.add_argument("--windows-size", type=int, default=128, help="The windows for inference")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Probability threshold for inference",
    )
    parser.add_argument("log-dir",type=str,help="The model load dir")
    args = parser.parse_args()

    infer(
        args,
    )