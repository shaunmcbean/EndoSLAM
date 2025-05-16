import open3d as o3d # pip3 install open3d numpy opencv-python
import numpy as np
import cv2
import os

#################
# CONFIGURATION #
#################

# data/
#    eval_frame/
#       image_0000.png
#       image_0001.png
#       ...
#    output_depth/
#       npy/
#           prediction_000.png
#           prediction_001.npy
#           prediction_002.npy
#           ...
#    camera_intrinsics.txt
#    camera_poses.csv

data_dir = './data'

# adjust according to data stucture
rgb_dir = os.path.join(data_dir, 'eval_frame') # rgb
depth_dir = os.path.join(data_dir, 'output_depth', 'npy') # .npy raw depth
intrinsics_file = os.path.join(data_dir, 'camera_intrinsics.txt') # camera intrinsic matrix
poses_file = os.path.join(data_dir,'camera_poses.csv') # camera pose ground truth EndoSLAM

# number of frames to process
num_frames_to_process = 400

def load_camera_intrinsics(filepath):
    
    try:
        intrinsics_data = np.loadtxt(filepath, delimiter=',')

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
            width=128,
            height=128,
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
        
        # including header
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        poses = []

        for row in data:

            tx, ty, tz, qx, qy, qz, qw = row[:7]

            # normalize quaternion
            norm = np.linalg.norm([qx, qy, qz, qw])
            if norm == 0:
                print(f"Warning: Zero norm quaternion encountered for pose {len(poses)}. Skipping pose or using identity.")
                R = np.eye(3)
            else:
                qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

                R = np.array([
                    [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
                    [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
                    [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
                ])

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            poses.append(T)

        print(f"Loaded {len(poses)} camera poses from {filepath}")
        return poses

    except Exception as e:
        print(f"Error loading poses from {filepath}: {e}")
        return None

def create_point_cloud_from_depth(rgb_image, depth_image, camera_intrinsics, camera_pose):

    # ensure depth image is float32 for Open3D
    if depth_image.dtype != np.float32:
         print(f"Warning: Converting depth image from {depth_image.dtype} to float32 for Open3D.")
         depth_image = depth_image.astype(np.float32)

    # convert images to Open3D Image format
    rgb_o3d = o3d.geometry.Image(rgb_image)
    depth_o3d = o3d.geometry.Image(depth_image)

    if np.asarray(rgb_o3d).shape[:2] != np.asarray(depth_o3d).shape[:2]:
        raise ValueError("RGB and depth image dimensions do not match")

    # adjust according to .npy depth data
    depth_scale = 1.0
    depth_trunc = 1.0
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,

        # true = b&w, false = color
        convert_rgb_to_intensity=False,
    )

    # update intrinsic matrix with actual image dimensions
    camera_intrinsics.width = rgb_image.shape[1]
    camera_intrinsics.height = rgb_image.shape[0]

    # create point cloud from RGB-D image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics,
    )

    # transform point cloud to global coordinate system using the camera pose
    pcd.transform(camera_pose)

    return pcd

##################
# RECONSTRUCTION #
##################

if __name__ == "__main__":

    # load camera intrinsics
    camera_intrinsics = load_camera_intrinsics(intrinsics_file)
    if camera_intrinsics:
        print(camera_intrinsics.intrinsic_matrix)
    else:
        print("Failed to load camera intrinsics. Exiting.")
        exit()


    # load camera poses
    camera_poses = load_camera_poses(poses_file)
    if camera_poses is None:
        print("Failed to load camera poses. Exiting.")
        exit()

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])


    if len(rgb_files) != len(depth_files) or len(rgb_files) == 0:
        print(f"Mismatch in the number of RGB ({len(rgb_files)}) and depth ({len(depth_files)}) files or no files found.")
        exit()

    # use the minimum of available frames, poses, or the desired num_frames_to_process
    max_frames = min(len(rgb_files), len(camera_poses), num_frames_to_process)

    # empty point cloud to accumulate reconstruction
    accumulated_point_cloud = o3d.geometry.PointCloud()

    print(f"Starting 3D reconstruction for the first {max_frames} frames...")

    for i in range(max_frames):
        rgb_filename = rgb_files[i]
        depth_filename = depth_files[i]

        rgb_path = os.path.join(rgb_dir, rgb_filename)
        depth_path = os.path.join(depth_dir, depth_filename)

        rgb_image = cv2.imread(rgb_path)

        # load raw depth data from .npy file
        try:
            depth_image = np.load(depth_path)
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

        # resize depth image to match RGB image dimensions
        target_height, target_width = rgb_image.shape[:2]
        depth_height, depth_width = depth_image.shape[:2]

        if depth_height != target_height or depth_width != target_width:
            print(f"Resizing depth image from ({depth_width}x{depth_height}) to match RGB ({target_width}x{target_height})")
            
            # interpolation. cv2.INTER_NEAREST for depth to avoid creating invalid depth values through averaging
            depth_image_resized = cv2.resize(depth_image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        else:
            depth_image_resized = depth_image 

        current_pose = camera_poses[i]

        # create and transform point cloud for the current frame
        pcd = create_point_cloud_from_depth(rgb_image, depth_image_resized, camera_intrinsics, current_pose)

        if pcd is not None:
            accumulated_point_cloud += pcd

        print(f"Processed frame {i+1}/{max_frames}")

    # downsampling and outlier removal for cleaner visualization
    print("Downsampling and removing outliers...")

    # adjust voxel size based on scene scale
    voxel_size = 0.001
    if len(accumulated_point_cloud.points) > 0:
        accumulated_point_cloud = accumulated_point_cloud.voxel_down_sample(voxel_size)

        # adjust nb_neighbors and std_ratio based on noise level
        cl, ind = accumulated_point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        accumulated_point_cloud = accumulated_point_cloud.select_by_index(ind)

    print(f"Final point cloud has {len(accumulated_point_cloud.points)} points.")
    print("3D reconstruction complete. Visualizing...")

    # visualization
    if len(accumulated_point_cloud.points) > 0:
        o3d.visualization.draw_geometries([accumulated_point_cloud])
    else:
        print("Accumulated point cloud is empty. Nothing to visualize.")

    ## save the reconstructed point cloud
    # if len(accumulated_point_cloud.points) > 0:
    #     o3d.io.write_point_cloud("endoslam_reconstruction.pcd", accumulated_point_cloud)