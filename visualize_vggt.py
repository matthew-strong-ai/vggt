import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
# from vggt.dependency.projection import get_camera_models

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


# Initialize the model and load the pretrained weights
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images
image_names = ["road_imgs/rgb_frame_0_original.png", "road_imgs/rgb_frame_1_original.png", 
               "road_imgs/rgb_frame_2_original.png", "road_imgs/rgb_frame_4_original.png", 
               ]  
images = load_and_preprocess_images(image_names).to(device)

print(f"Loaded {len(image_names)} images")

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps
        predictions, tokens_list = model(images)

print("Model outputs:", predictions.keys())

# HOW TO USE FEATURES
# tokens_list is a list of aggregated tokens for intermediate features
print(f"Number of token sets: {len(tokens_list)}")

# get the shape of the last set of tokens, S is the number of input images
last_tokens = tokens_list[-1]
print(f"Shape of last tokens: {last_tokens.shape}")  # Expected shape: [B, S, num_tokens, token_dim]



# Create output directory
os.makedirs("vggt_outputs", exist_ok=True)

# Visualize depth maps
if "depth" in predictions:
    depth_maps = predictions["depth"].cpu().numpy()
    depth_conf = predictions["depth_conf"].cpu().numpy()
    
    # Create figure for depth visualization
    fig, axes = plt.subplots(2, len(image_names), figsize=(20, 8))
    fig.suptitle("VGGT Depth Predictions", fontsize=16)
    
    for i in range(len(image_names)):
        # Show original image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Frame {i}")
        axes[0, i].axis('off')
        
        # Show depth map
        depth = depth_maps[0, i, :, :, 0]
        depth_vis = axes[1, i].imshow(depth, cmap='viridis')
        axes[1, i].set_title(f"Depth {i}")
        axes[1, i].axis('off')
    
    plt.colorbar(depth_vis, ax=axes[1, :], label='Depth')
    plt.tight_layout()
    plt.savefig("vggt_outputs/depth_visualization.png", dpi=150)
    print("Saved depth visualization to vggt_outputs/depth_visualization.png")
    plt.close()

# Visualize camera poses
# if "pose_enc" in predictions:
#     pose_enc = predictions["pose_enc"].cpu()
    
#     # Convert pose encoding to extrinsic matrices
#     extrinsics, intrinsics = get_camera_models(pose_enc, width=images.shape[-1], height=images.shape[-2])
    
#     # Visualize camera positions in 3D
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Extract camera positions (last column of extrinsic matrix)
#     camera_positions = []
#     for i in range(extrinsics.shape[1]):
#         # Camera position is -R^T @ t where R is rotation and t is translation
#         R = extrinsics[0, i, :3, :3].numpy()
#         t = extrinsics[0, i, :3, 3].numpy()
#         cam_pos = -R.T @ t
#         camera_positions.append(cam_pos)
        
#         # Plot camera
#         ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], s=100, c='red')
#         ax.text(cam_pos[0], cam_pos[1], cam_pos[2], f'  Cam{i}', fontsize=8)
        
#         # Draw camera viewing direction
#         z_axis = R.T @ np.array([0, 0, 1])
#         ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], 
#                  z_axis[0], z_axis[1], z_axis[2], 
#                  length=0.5, color='blue', arrow_length_ratio=0.1)
    
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Camera Poses')
#     plt.savefig("vggt_outputs/camera_poses.png", dpi=150)
#     print("Saved camera poses to vggt_outputs/camera_poses.png")
#     plt.close()

# Visualize 3D point cloud
if "world_points" in predictions:
    world_points = predictions["world_points"].cpu().numpy()
    world_points_conf = predictions["world_points_conf"].cpu().numpy()
    
    # Sample points for visualization (too many points to visualize all)
    sample_rate = 50  # Sample every 50th point
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Collect points from all frames
    all_points = []
    all_colors = []
    
    for i in range(len(image_names)):
        pts = world_points[0, i, ::sample_rate, ::sample_rate, :].reshape(-1, 3)
        conf = world_points_conf[0, i, ::sample_rate, ::sample_rate].reshape(-1)
        img_colors = images[i].cpu().permute(1, 2, 0).numpy()[::sample_rate, ::sample_rate, :].reshape(-1, 3)
        
        # Filter by confidence
        mask = conf > np.percentile(conf, 50)  # Keep top 50% confident points
        pts = pts[mask]
        img_colors = img_colors[mask]
        
        all_points.append(pts)
        all_colors.append(img_colors)
    
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    
    # Plot point cloud
    ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], 
               c=all_colors, s=1, alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')
    ax.view_init(elev=30, azim=45)
    
    plt.savefig("vggt_outputs/point_cloud.png", dpi=150)
    print("Saved point cloud to vggt_outputs/point_cloud.png")
    plt.close()

# Create a summary figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("VGGT Results Summary", fontsize=16)

# Show first three frames
for i in range(3):
    img = images[i].cpu().permute(1, 2, 0).numpy()
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"Input Frame {i}")
    axes[0, i].axis('off')
    
    if "depth" in predictions:
        depth = depth_maps[0, i, :, :, 0]
        axes[1, i].imshow(depth, cmap='viridis')
        axes[1, i].set_title(f"Depth Map {i}")
        axes[1, i].axis('off')

plt.tight_layout()
plt.savefig("vggt_outputs/summary.png", dpi=150)
print("Saved summary to vggt_outputs/summary.png")

print("\nVisualization complete! Check the vggt_outputs/ folder for results.")