import os
from threading import Thread
from SplaTAM.datasets.gradslam_datasets.geometryutils import relative_transformation
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_msssim import ms_ssim

from SplaTAM.utils.slam_external import calc_psnr
from SplaTAM.utils.eval_helpers import evaluate_ate, loss_fn_alex

from gsplatam.geometry import build_transform
from gsplatam.renderer import Camera


def rotation_error(gt_w2c_list, est_w2c_list):
    """
    Compute the rotation error between the estimated and ground truth camera poses in degrees.
    """
    errors = []
    for gt_w2c, est_w2c in zip(gt_w2c_list, est_w2c_list):
        gt_rot = gt_w2c[:3, :3]
        est_rot = est_w2c[:3, :3]
        gt_rot_inv = torch.linalg.inv(gt_rot)
        relative_rot = torch.matmul(est_rot, gt_rot_inv)
        angle = torch.acos(torch.clamp((torch.trace(relative_rot) - 1) / 2, -1, 1))
        angle_deg = torch.rad2deg(angle)
        errors.append(angle_deg.item())
    return errors


def plot_rgbd_silhouette(color, depth, rastered_color, rastered_depth, presence_sil_mask, diff_depth_l1,
                         psnr, depth_l1, fig_title, plot_dir=None, plot_name=None, 
                         save_plot=False, wandb_run=None, wandb_step=None, wandb_title=None, diff_rgb=None):
    C, H, W = color.shape
    rastered_color = torch.clamp(rastered_color, 0, 1)
    H = H // 2
    W = W // 2

    max_depth = 6
    figure = np.zeros((H * 2, W * 3, 3), dtype=np.uint8)
    figure[:H, :W, :] = color.cpu().permute(1, 2, 0).numpy()[::2, ::2] * 255
    figure[:H, W:2*W, :] = depth[0, ::2, ::2, None].cpu().numpy() / max_depth * 255
    figure[:H, 2*W:3*W, :] = presence_sil_mask[::2, ::2, None] * 255
    figure[H:2*H, :W, :] = rastered_color.cpu().permute(1, 2, 0).numpy()[::2, ::2] * 255
    figure[H:2*H, W:2*W, :] = rastered_depth[0, ::2, ::2, None].cpu().numpy() / max_depth * 255
    figure[H:2*H, 2*W:3*W, :] = diff_depth_l1[0, ::2, ::2, None].cpu().numpy() / max_depth * 255

    # write labels with cv2
    # make black border
    cv2.putText(figure, 'Ground Truth RGB', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(figure, "Ground Truth RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(figure, 'Ground Truth Depth', (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(figure, "Ground Truth Depth", (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(figure, 'Rasterized Silhouette', (2*W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(figure, "Rasterized Silhouette", (2*W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(figure, 'Rasterized RGB', (10, H + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(figure, "Rasterized RGB", (10, H + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(figure, 'Rasterized Depth', (W + 10, H + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(figure, "Rasterized Depth", (W + 10, H + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(figure, 'Diff Depth L1', (2*W + 10, H + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(figure, "Diff Depth L1", (2*W + 10, H + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(figure, "PSNR: {:.2f}".format(psnr), (10, H + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(figure, "PSNR: {:.2f}".format(psnr), (10, H + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(figure, "L1: {:.4f}".format(depth_l1), (W + 10, H + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(figure, "L1: {:.4f}".format(depth_l1), (W + 10, H + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(plot_dir, f"{plot_name}.png"), figure[..., ::-1])
    
    if wandb_run is not None:
        if wandb_step is None:
            wandb_run.log({wandb_title: figure})
        else:
            wandb_run.log({wandb_title: figure}, step=wandb_step)


@torch.no_grad()
def eval(
    render_fn,
    dataset, final_params, num_frames, eval_dir, sil_thres, 
    mapping_iters, add_new_gaussians, wandb_run=None, wandb_save_qual=False, eval_every=1, save_frames=False
):
    print("Evaluating Final Parameters ...")
    psnr_list = []
    rmse_list = []
    l1_list = []
    lpips_list = []
    ssim_list = []
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    if save_frames:
        render_rgb_dir = os.path.join(eval_dir, "rendered_rgb")
        os.makedirs(render_rgb_dir, exist_ok=True)
        render_depth_dir = os.path.join(eval_dir, "rendered_depth")
        os.makedirs(render_depth_dir, exist_ok=True)
        rgb_dir = os.path.join(eval_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        depth_dir = os.path.join(eval_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)
    
    dataset.device = 'cpu'
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    dataloader_iter = dataloader.__iter__()

    gt_w2c_list = []
    threads = []
    for time_idx in tqdm(range(num_frames)):
         # Get RGB-D Data & Camera Parameters
        color, depth, intrinsics, pose = next(dataloader_iter)
        color, depth, intrinsics, pose = color[0].cuda(), depth[0].cuda(), intrinsics[0].cuda(), pose[0].cuda()
        # color, depth, intrinsics, pose = dataset[time_idx]
        # color, depth, intrinsics, pose = color.cuda(), depth.cuda(), intrinsics.cuda(), pose.cuda()
        gt_w2c = torch.linalg.inv(pose)
        gt_w2c_list.append(gt_w2c)
        intrinsics = intrinsics[:3, :3]

        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

        if time_idx == 0:
            # Process Camera Parameters
            first_frame_w2c = torch.linalg.inv(pose)
            # Setup Camera
            cam = Camera(intrinsics[None], color.shape[2], color.shape[1])
        
        # Skip frames if not eval_every
        if time_idx != 0 and (time_idx+1) % eval_every != 0:
            continue
 
        # Define current frame data
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': time_idx, 'w2c': first_frame_w2c}

        # Render Depth & Silhouette
        im, rastered_depth, silhouette = render_fn(cam, final_params, time_idx, False, False)
        # Mask invalid depth in GT
        valid_depth_mask = (curr_data['depth'] > 0)
        rastered_depth_viz = rastered_depth.detach()
        rastered_depth = rastered_depth * valid_depth_mask
        presence_sil_mask = (silhouette > sil_thres)[0]
        
        # Render RGB and Calculate PSNR
        if mapping_iters==0 and not add_new_gaussians:
            weighted_im = im * presence_sil_mask * valid_depth_mask
            weighted_gt_im = curr_data['im'] * presence_sil_mask * valid_depth_mask
        else:
            weighted_im = im * valid_depth_mask
            weighted_gt_im = curr_data['im'] * valid_depth_mask
        psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
        ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                        data_range=1.0, size_average=True)
        lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                    torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

        psnr_list.append(psnr.cpu().numpy())
        ssim_list.append(ssim.cpu().numpy())
        lpips_list.append(lpips_score)

        # Compute Depth RMSE
        if mapping_iters==0 and not add_new_gaussians:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - curr_data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - curr_data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        rmse_list.append(rmse.cpu().numpy())
        l1_list.append(depth_l1.cpu().numpy())

        if save_frames:
            # Save Rendered RGB and Depth
            viz_render_im = torch.clamp(im, 0, 1)
            viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
            vmin = 0
            vmax = 6
            viz_render_depth = rastered_depth_viz[0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_render_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(render_rgb_dir, "gs_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_render_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(render_depth_dir, "gs_{:04d}.png".format(time_idx)), depth_colormap)

            # Save GT RGB and Depth
            viz_gt_im = torch.clamp(curr_data['im'], 0, 1)
            viz_gt_im = viz_gt_im.detach().cpu().permute(1, 2, 0).numpy()
            viz_gt_depth = curr_data['depth'][0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_gt_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(rgb_dir, "gt_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_gt_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(depth_dir, "gt_{:04d}.png".format(time_idx)), depth_colormap)
        
        # Plot the Ground Truth and Rasterized RGB & Depth, along with Silhouette
        fig_title = "Time Step: {}".format(time_idx)
        plot_name = "%04d" % time_idx
        presence_sil_mask = presence_sil_mask.detach().cpu().numpy()
        if wandb_run is None:
            threads.append(Thread(target=plot_rgbd_silhouette,
                   args=(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
                         psnr, depth_l1, fig_title, plot_dir, plot_name, True)
            ))
            threads[-1].start()
            # plot_rgbd_silhouette(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
            #                      psnr, depth_l1, fig_title, plot_dir, 
            #                      plot_name=plot_name, save_plot=True)
        elif wandb_save_qual:
            plot_rgbd_silhouette(color, depth, im, rastered_depth_viz, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, plot_dir, 
                                 plot_name=plot_name, save_plot=True,
                                 wandb_run=wandb_run, wandb_step=None, 
                                 wandb_title="Eval/Qual Viz")
    [t.join() for t in threads]

    # Compute the final ATE RMSE
    # Get the final camera trajectory
    num_frames = final_params['cam_unnorm_rots'].shape[-1]
    latest_est_w2c = first_frame_w2c
    latest_est_w2c_list = []
    latest_est_w2c_list.append(latest_est_w2c)
    valid_gt_w2c_list = []
    valid_gt_w2c_list.append(gt_w2c_list[0])
    for idx in range(1, num_frames):
        # Check if gt pose is not nan for this time step
        if torch.isnan(gt_w2c_list[idx]).sum() > 0:
            continue
        intermrel_w2c = build_transform(
            final_params['cam_trans'][idx].detach(),
            final_params['cam_unnorm_rots'][idx].detach()
        )
        latest_est_w2c = intermrel_w2c
        latest_est_w2c_list.append(latest_est_w2c)
        valid_gt_w2c_list.append(gt_w2c_list[idx])
    gt_w2c_list = valid_gt_w2c_list
    # Calculate ATE RMSE
    ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
    mean_rotation_error_deg = rotation_error(gt_w2c_list, latest_est_w2c_list)
    print("Final Average ATE RMSE: {:.2f} cm".format(ate_rmse*100))
    print("Final Average Rotation Error: {:.2f} degrees".format(np.mean(mean_rotation_error_deg)))
    if wandb_run is not None:
        wandb_run.log({"Final Stats/Avg ATE RMSE": ate_rmse,
                    "Final Stats/step": 1})
    
    # Compute Average Metrics
    psnr_list = np.array(psnr_list)
    rmse_list = np.array(rmse_list)
    l1_list = np.array(l1_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    avg_psnr = psnr_list.mean()
    avg_rmse = rmse_list.mean()
    avg_l1 = l1_list.mean()
    avg_ssim = ssim_list.mean()
    avg_lpips = lpips_list.mean()
    print("Average PSNR: {:.2f}".format(avg_psnr))
    print("Average Depth RMSE: {:.2f} cm".format(avg_rmse*100))
    print("Average Depth L1: {:.2f} cm".format(avg_l1*100))
    print("Average MS-SSIM: {:.3f}".format(avg_ssim))
    print("Average LPIPS: {:.3f}".format(avg_lpips))

    if wandb_run is not None:
        wandb_run.log({"Final Stats/Average PSNR": avg_psnr, 
                       "Final Stats/Average Depth RMSE": avg_rmse,
                       "Final Stats/Average Depth L1": avg_l1,
                       "Final Stats/Average MS-SSIM": avg_ssim, 
                       "Final Stats/Average LPIPS": avg_lpips,
                       "Final Stats/step": 1})

    # Save metric lists as text files
    np.savetxt(os.path.join(eval_dir, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(eval_dir, "rmse.txt"), rmse_list)
    np.savetxt(os.path.join(eval_dir, "l1.txt"), l1_list)
    np.savetxt(os.path.join(eval_dir, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(eval_dir, "lpips.txt"), lpips_list)

    # Plot PSNR & L1 as line plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(np.arange(len(psnr_list)), psnr_list)
    axs[0].set_title("RGB PSNR")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("PSNR")
    axs[1].plot(np.arange(len(l1_list)), l1_list*100)
    axs[1].set_title("Depth L1")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("L1 (cm)")
    fig.suptitle("Average PSNR: {:.2f}, Average Depth L1: {:.2f} cm, ATE RMSE: {:.2f} cm".format(avg_psnr, avg_l1*100, ate_rmse*100), y=1.05, fontsize=16)
    plt.savefig(os.path.join(eval_dir, "metrics.png"), bbox_inches='tight')
    if wandb_run is not None:
        wandb_run.log({"Eval/Metrics": fig})
    plt.close()

