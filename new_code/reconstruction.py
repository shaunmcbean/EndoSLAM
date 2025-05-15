import open3d as o3d # pip3 install open3d numpy opencv-python
import numpy as np
import cv2
import os

# --- Configuration ---
# Replace with the actual paths to your data
# This example assumes a simple structure:
# data_dir/
# |- seq_01/
#    |- rgb/
#    |  |- frame_00000.png
#    |  |- frame_00001.png
#    |  |  ...
#    |- depth/
#    |  |- raw_depth_00000.npy # Raw predicted depth maps (e.g., from your model, saved as .npy)
#    |  |- raw_depth_00001.npy
#    |  |  ...
#    |- camera_intrinsics.txt # File containing camera intrinsic matrix
#    |- camera_poses.csv      # File containing camera poses (e.g., ground truth from EndoSLAM)

data_dir = './data'

rgb_dir = os.path.join(data_dir, 'eval_frame') #rgb
depth_dir = os.path.join(data_dir, 'output_depth', 'output_raw') #directory where you saved the raw_depth_*.npy files
intrinsics_file = os.path.join(data_dir, 'camera_intrinsics.txt') #camera intrinsic matrix
poses_file = os.path.join(data_dir,'camera_poses.csv') #camera poses

# Number of frames to process (adjust based on your sequence length)
num_frames_to_process = 50

# --- Helper Functions ---

def load_camera_intrinsics(filepath):
    """Loads camera intrinsic matrix from a text file."""
    try:
        # Load the data from the file
        intrinsics_data = np.loadtxt(filepath, delimiter=',') # Specify delimiter if comma-separated

        # Check if the data is a flattened array (shape (9,)) and reshape it
        if intrinsics_data.shape == (9,):
            intrinsics = intrinsics_data.reshape(3, 3)
            print(f"Loaded flattened intrinsics from {filepath} and reshaped to 3x3.")
        elif intrinsics_data.shape == (3, 3):
            intrinsics = intrinsics_data
            print(f"Loaded 3x3 intrinsics from {filepath}.")
        else:
            raise ValueError(f"Intrinsic data has unexpected shape: {intrinsics_data.shape}. Expected (9,) or (3,3).")

        # Ensure the data type is float for Open3D
        intrinsics = intrinsics.astype(np.float64)

        return o3d.camera.PinholeCameraIntrinsic(
            width=0, # Placeholder, actual width/height should match images
            height=0, # Placeholder
            fx=intrinsics[0, 0],
            fy=intrinsics[1, 1],
            cx=intrinsics[0, 2],
            cy=intrinsics[1, 2]
        )
    except Exception as e:
        print(f"Error loading intrinsics from {filepath}: {e}")
        return None

def load_camera_poses(filepath):
    """Loads camera poses from a CSV with format: tX, tY, tZ, rX, rY, rZ, rW"""
    try:
        # Adjust skiprows if your CSV has a different header
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        poses = []

        for row in data:
            # Assuming the first 7 columns are tx, ty, tz, qx, qy, qz, qw
            tx, ty, tz, qx, qy, qz, qw = row[:7]

            # Normalize quaternion (important for valid rotation matrix)
            norm = np.linalg.norm([qx, qy, qz, qw])
            if norm == 0:
                print(f"Warning: Zero norm quaternion encountered for pose {len(poses)}. Skipping pose or using identity.")
                R = np.eye(3) # Or handle as an error
            else:
                qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

                # Compute rotation matrix from quaternion
                R = np.array([
                    [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
                    [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
                    [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
                ])

            # Assemble into 4x4 transformation matrix (camera to world/global)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            poses.append(T)

        print(f"Loaded {len(poses)} camera poses from {filepath}")
        return poses

    except Exception as e:
        print(f"Error loading poses from {filepath}: {e}")
        return None

# def convert_heatmap_to_depth(heatmap_uint8, min_depth=0.1, max_depth=5.0):
#     # This function is likely not needed if you save raw depth
#     normalized = heatmap_uint8.astype(np.float32) / 255.0
#     depth = normalized * (max_depth - min_depth) + min_depth
#     return depth


def create_point_cloud_from_depth(rgb_image, depth_image, camera_intrinsics, camera_pose):
    """
    Creates a colored point cloud from an RGB image and depth map
    and transforms it to the global coordinate system.
    Assumes depth_image is a numpy array of metric depth values (e.g., in meters).
    """
    # Ensure depth image is float32 for Open3D
    if depth_image.dtype != np.float32:
         print(f"Warning: Converting depth image from {depth_image.dtype} to float32 for Open3D.")
         depth_image = depth_image.astype(np.float32)

    # Convert images to Open3D Image format
    rgb_o3d = o3d.geometry.Image(rgb_image)
    depth_o3d = o3d.geometry.Image(depth_image)

    # Check for dimension mismatch before creating RGBDImage
    if np.asarray(rgb_o3d).shape[:2] != np.asarray(depth_o3d).shape[:2]:
        raise ValueError("RGB and depth image dimensions do not match")



    # Create an RGB-D image
    # depth_scale=1.0 assumes your depth_image is already in meters
    # depth_trunc should be set based on the expected range of valid depth in your scene
    depth_scale = 1.0 # Assuming your .npy depth values are already in meters
    depth_trunc = 5.0    # Example: Truncate depth beyond 5 meters (adjust based on endoscopic scene)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False # Keep color
    )

    # Update intrinsic matrix with actual image dimensions (important after loading)
    camera_intrinsics.width = rgb_image.shape[1]
    camera_intrinsics.height = rgb_image.shape[0]

    # Create point cloud from RGB-D image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics,
        # Set stride if you want to skip pixels (e.g., stride=2 for half resolution)
        # stride=1
    )

    # Transform the point cloud to the global coordinate system using the camera pose
    # Assuming camera_pose is a 4x4 transformation matrix from camera to world
    pcd.transform(camera_pose)

    return pcd

# --- Main Reconstruction Process ---

if __name__ == "__main__":
    # Load camera intrinsics
    camera_intrinsics = load_camera_intrinsics(intrinsics_file)
    if camera_intrinsics:
        print(camera_intrinsics.intrinsic_matrix)
    else:
        print("Failed to load camera intrinsics. Exiting.")
        exit()


    # Load camera poses
    camera_poses = load_camera_poses(poses_file)
    if camera_poses is None:
        print("Failed to load camera poses. Exiting.")
        exit()

    # Get list of image and depth files
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    # Assuming you saved depth as .npy
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])


    if len(rgb_files) != len(depth_files) or len(rgb_files) == 0:
        print(f"Mismatch in the number of RGB ({len(rgb_files)}) and depth ({len(depth_files)}) files or no files found.")
        exit()

    # Use the minimum of available frames, poses, or the desired num_frames_to_process
    max_frames = min(len(rgb_files), len(camera_poses), num_frames_to_process)


    # Create an empty point cloud to accumulate the reconstruction
    accumulated_point_cloud = o3d.geometry.PointCloud()

    print(f"Starting 3D reconstruction for the first {max_frames} frames...")

    for i in range(max_frames):
        rgb_filename = rgb_files[i]
        depth_filename = depth_files[i] # Assuming sorted lists correspond

        rgb_path = os.path.join(rgb_dir, rgb_filename)
        depth_path = os.path.join(depth_dir, depth_filename)


        # Load RGB image (cv2 loads as BGR by default, Open3D expects RGB, but for visualization it often handles BGR)
        # To be precise for texturing later, might need conversion cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.imread(rgb_path)

        # Load raw depth data from .npy file
        try:
            depth_image = np.load(depth_path)
            # Ensure depth is loaded as float32 if it's not already
            if depth_image.dtype != np.float32:
                 depth_image = depth_image.astype(np.float32)

        except Exception as e:
            print(f"Error loading depth data from {depth_path}: {e}")
            continue


        if rgb_image is None:
            print(f"Error loading RGB image: {rgb_path}")
            continue
        if depth_image is None:
            print(f"Error loading Depth data from {depth_path}")
            continue

        # --- IMPORTANT: Resize depth image to match RGB image dimensions ---
        # Get the target dimensions from the loaded RGB image
        target_height, target_width = rgb_image.shape[:2]
        depth_height, depth_width = depth_image.shape[:2]

        if depth_height != target_height or depth_width != target_width:
            print(f"Resizing depth image from ({depth_width}x{depth_height}) to match RGB ({target_width}x{target_height})")
            # Use interpolation. cv2.INTER_NEAREST is often preferred for depth
            # to avoid creating invalid depth values through averaging.
            depth_image_resized = cv2.resize(depth_image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        else:
            depth_image_resized = depth_image # Dimensions already match


        # Get the camera pose for the current frame
        current_pose = camera_poses[i]

        # Create and transform point cloud for the current frame
        # Pass the resized depth image
        pcd = create_point_cloud_from_depth(rgb_image, depth_image_resized, camera_intrinsics, current_pose)

        if pcd is not None:
            # Add the point cloud to the accumulated reconstruction
            accumulated_point_cloud += pcd

        print(f"Processed frame {i+1}/{max_frames}")

    # --- Optional: Downsampling and outlier removal for cleaner visualization ---
    print("Downsampling and removing outliers...")
    # Voxel downsampling
    # Adjust voxel size based on scene scale (e.g., 0.005 meters)
    voxel_size = 0.005
    if len(accumulated_point_cloud.points) > 0:
        accumulated_point_cloud = accumulated_point_cloud.voxel_down_sample(voxel_size)

        # Statistical outlier removal
        # Adjust nb_neighbors and std_ratio based on noise level
        cl, ind = accumulated_point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        accumulated_point_cloud = accumulated_point_cloud.select_by_index(ind)

    print(f"Final point cloud has {len(accumulated_point_cloud.points)} points.")
    print("3D reconstruction complete. Visualizing...")

    # --- Visualization ---
    if len(accumulated_point_cloud.points) > 0:
        o3d.visualization.draw_geometries([accumulated_point_cloud])
    else:
        print("Accumulated point cloud is empty. Nothing to visualize.")

    # You can save the reconstructed point cloud
    # if len(accumulated_point_cloud.points) > 0:
    #     o3d.io.write_point_cloud("endoslam_reconstruction.pcd", accumulated_point_cloud)