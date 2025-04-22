import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
import imageio


def draw_camera(ax, pose, color='r', scale=0.05):
    R = pose[:3, :3]
    t = pose[:3, 3]
    
    x_axis = R[:, 0] * scale
    y_axis = R[:, 1] * scale
    z_axis = R[:, 2] * scale

    ax.quiver(*t, *x_axis, color='r')  # X
    ax.quiver(*t, *y_axis, color='g')  # Y
    ax.quiver(*t, *z_axis, color='b')  # Z

def visualize_pose(stats, save_path=None):
    """
    Visualize the pose statistics.

    Args:
        stats (dict): Dictionary containing pose statistics.
        save_path (str, optional): Path to save the plot. If None, the plot will be shown instead.
    """
    gt_poses = stats["pose"]["gt_pose"]
    opt_poses = stats["pose"]["opt_pose"]
    
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i in range(len(gt_poses)):
        T = opt_poses[i]
        T_gt = gt_poses[i]
    
        R = T[:3, :3]
        t = T[:3, 3]
        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        origin_gt = t_gt
        x_axis_gt = R_gt[:, 0]
        y_axis_gt = R_gt[:, 1]
        z_axis_gt = R_gt[:, 2]
        ax.quiver(*origin_gt, *x_axis_gt, color='r', label='X GT')
        ax.quiver(*origin_gt, *y_axis_gt, color='g', label='Y GT')
        ax.quiver(*origin_gt, *z_axis_gt, color='b', label='Z GT')

        origin = t
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]

        ax.quiver(*origin, *x_axis, color='r', linestyle='dashed', label='X local')
        ax.quiver(*origin, *y_axis, color='g', linestyle='dashed', label='Y local')
        ax.quiver(*origin, *z_axis, color='b', linestyle='dashed', label='Z local')

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('SE(3) Transform Visualization')
        plt.legend()
        path = os.path.join(save_dir, f"pose_visualization_{i}.png")
        plt.savefig(path)
        plt.show()
        plt.close()
    
    return

def create_pose_gif(image_dir, output_path, duration=0.5):
    """
    Create a GIF from pose visualization images.

    Args:
        image_dir (str): Directory containing pose visualization images.
        output_path (str): Path to save the resulting GIF.
        duration (float): Duration between frames in seconds.
    """
    images = []
    filenames = sorted([f for f in os.listdir(image_dir) if f.startswith("pose_visualization_") and f.endswith(".png")],
                       key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for filename in filenames:
        image_path = os.path.join(image_dir, filename)
        images.append(imageio.imread(image_path))
    imageio.mimsave(output_path, images, duration=duration) 

if __name__ == "__main__":
    # stats_pth = "C:\Workspace\Codebase\\neuralfeels\outputs\\2025-04-14\\13-25-23\\077_rubiks_cube\\00\\vitac\pose\stats.pkl"
    # with open(stats_pth, "rb") as f:
    #     stats = pickle.load(f)
    image_dir = r"C:\Workspace\\Codebase\\neuralfeels\\outputs\\2025-04-16\\10-56-03\\077_rubiks_cube\\00\\vitac\\pose_1"
    output_gif = os.path.join(image_dir, "pose_animation.gif")
    create_pose_gif(image_dir, output_gif)