import torch
from detectron2.structures import Boxes
import numpy as np
import matplotlib.pyplot as plt
#import open3d as o3d

from detectron2.structures import pairwise_iou
from pytorch3d.ops import box3d_overlap

##### Proposal
def normalize_vector(v):
    v_mag = torch.sqrt(v.pow(2).sum())
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])))
    v_mag = v_mag.view(1,1).expand(1,v.shape[0])
    v = v/v_mag

    return v[0]
    
def cross_product(u, v):
    i = u[1]*v[2] - u[2]*v[1]
    j = u[2]*v[0] - u[0]*v[2]
    k = u[0]*v[1] - u[1]*v[0]
    out = torch.cat((i.view(1,1), j.view(1,1), k.view(1,1)),1)
        
    return out[0]

def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[0:3]
    y_raw = poses[3:6]
        
    x = normalize_vector(x_raw)
    z = cross_product(x,y_raw)
    z = normalize_vector(z)
    y = cross_product(z,x)
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2)[0]

    return matrix

def sample_normal_in_range(means, stds, count, threshold_low=None, threshold_high=None):
    device = means.device
    # Generate samples from a normal distribution
    if len(means.size()) == 0:
        samples = torch.normal(means, stds, size=(count,))
    else:
        samples = torch.normal(means.unsqueeze(1).expand(-1,count), stds.unsqueeze(1).expand(-1,count))

    # Ensure that all samples are greater than threshold_low and less than threshold_high
    if not (threshold_high or threshold_low is None):
        while torch.any((samples < threshold_low) | (samples > threshold_high)): # TODO stop argument in case of never sampling
            invalid_mask = (samples < threshold_low) | (samples > threshold_high)
            # Replace invalid samples with new samples drawn from the normal distribution
            samples[invalid_mask] = torch.normal(means, stds, size=(invalid_mask.sum(),))

    return samples.to(device)

def randn_orthobasis_torch(num_samples=1,num_instances=1):
    z = torch.randn(num_instances, num_samples, 3, 3)
    z = z / torch.norm(z, p=2, dim=-1, keepdim=True)
    z[:, :, 0] = torch.cross(z[:, :, 1], z[:, :, 2], dim=-1)
    z[:, :, 0] = z[:, :, 0] / torch.norm(z[:, :, 0], dim=-1, keepdim=True)
    z[:, :, 1] = torch.cross(z[:, :, 2], z[:, :, 0], dim=-1)
    z[:, :, 1] = z[:, :, 1] / torch.norm(z[:, :, 1], dim=-1, keepdim=True)
    return z

def randn_orthobasis(num_samples=1):
    z = np.random.randn(num_samples, 3, 3)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    z[:, 0] = np.cross(z[:, 1], z[:, 2], axis=-1)
    z[:, 0] = z[:, 0] / np.linalg.norm(z[:, 0], axis=-1, keepdims=True)
    z[:, 1] = np.cross(z[:, 2], z[:, 0], axis=-1)
    z[:, 1] = z[:, 1] / np.linalg.norm(z[:, 1], axis=-1, keepdims=True)
    return z

"""
# plotting
import open3d as o3d
def draw_vector(vector, color=(0, 0, 1)):
    # Create a LineSet object
    line_set = o3d.geometry.LineSet()

    # Set the points of the LineSet to be the origin and the vector
    line_set.points = o3d.utility.Vector3dVector([np.zeros(3), vector])
    line_set.colors = o3d.utility.Vector3dVector([color, color])

    # Set the lines of the LineSet to be a line from the first point to the second point
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])

    # Draw the LineSet
    return line_set

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
# display normal vector in point cloud 
plane = pcd.select_by_index(best_inliers).paint_uniform_color([1, 0, 0])
not_plane = pcd.select_by_index(best_inliers, invert=True)
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])
# X-axis : Red arrow
# Y-axis : Green arrow
# Z-axis : Blue arrow
obb = plane.get_oriented_bounding_box()
obb.color = [0, 0, 1]
objs = [plane, not_plane, mesh, obb, utils.draw_vector(normal_vec)]
o3d.visualization.draw_geometries(objs)
"""

# ##things for making rotations
def vec_perp(vec):
    '''generate a vector perpendicular to vec in 3d'''
    # https://math.stackexchange.com/a/2450825
    a, b, c = vec
    if a == 0:
        return np.array([0,c,-b])
    return np.array(normalize_vector(torch.tensor([b,-a,0])))

def orthobasis_from_normal(normal, yaw_angle=0):
    '''generate an orthonormal/Rotation matrix basis from a normal vector in 3d
     
       returns a 3x3 matrix with the basis vectors as columns, 3rd column is the original normal vector
    '''
    x = rotate_vector(vec_perp(normal), normal, yaw_angle)
    x = x / np.linalg.norm(x, ord=2)
    y = np.cross(normal, x)
    return np.array([x, normal, y]).T # the vectors should be as columns

def rotate_vector(v, k, theta):
    '''rotate a vector v around an axis k by an angle theta
    it is assumed that k is a unit vector (p2 norm = 1)'''
    # https://medium.com/@sim30217/rodrigues-rotation-formula-47489db49050
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    term1 = v * cos_theta
    term2 = np.cross(k, v) * sin_theta
    term3 = k * np.dot(k, v) * (1 - cos_theta)
    
    return term1 + term2 + term3

def vec_perp_t(vec):
    '''generate a vector perpendicular to vec in 3d'''
    # https://math.stackexchange.com/a/2450825
    a, b, c = vec
    if a == 0:
        return torch.tensor([0,c,-b])
    return normalize_vector(torch.tensor([b,-a,0]))

def orthobasis_from_normal_t(normal:torch.Tensor, yaw_angles:torch.Tensor=0):
    '''generate an orthonormal/Rotation matrix basis from a normal vector in 3d

        normal is assumed to be normalised 
     
       returns a (no. of yaw_angles)x3x3 matrix with the basis vectors as columns, 3rd column is the original normal vector
    '''
    n = len(yaw_angles)
    x = rotate_vector_t(vec_perp_t(normal), normal, yaw_angles)
    # x = x / torch.norm(x, p=2)
    y = torch.cross(normal.view(-1,1), x)
    # y = y / torch.norm(y, p=2, dim=1)
    return torch.cat([x.t(), normal.unsqueeze(0).repeat(n, 1), y.t()],dim=1).reshape(n,3,3).transpose(2,1) # the vectors should be as columns

def rotate_vector_t(v, k, theta):
    '''rotate a vector v around an axis k by an angle theta
    it is assumed that k is a unit vector (p2 norm = 1)'''
    # https://medium.com/@sim30217/rodrigues-rotation-formula-47489db49050
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    v2 = v.view(-1,1)

    term1 = v2 * cos_theta
    term2 = torch.cross(k, v).view(-1, 1) * sin_theta
    term3 = (k * (k @ v)).view(-1, 1) * (1 - cos_theta)
    
    return (term1 + term2 + term3)

# ########### End rotations
def gt_in_norm_range(range,gt):
    if range[0] > 0: # both positive
        tmp = gt-range[0]
        res = tmp / abs(range[1] - range[0])
    elif range[1] > 0: # lower negative upper positive
        if gt > 0:
            tmp = gt-range[0]
        else:
            tmp = range[1]-gt
        res = tmp / abs(range[1] - range[0])
    else: # both negative
        tmp = range[1]-gt
        res = tmp / abs(range[1] - range[0])

    return res

def vectorized_linspace(start_tensor, end_tensor, number_of_steps):
    # Calculate spacing
    spacing = (end_tensor - start_tensor) / (number_of_steps - 1)
    # Create linear spaces with arange
    linear_spaces = torch.arange(start=0, end=number_of_steps, dtype=start_tensor.dtype)
    linear_spaces = linear_spaces.repeat(start_tensor.size(0),1)
    linear_spaces = linear_spaces * spacing[:,None] + start_tensor[:,None]
    return linear_spaces







##### Scoring
def iou_2d(gt_box, proposal_boxes):
    '''
    gt_box: Boxes
    proposal_box: Boxes
    '''
    IoU = pairwise_iou(gt_box,proposal_boxes).flatten()
    return IoU

def iou_3d(gt_cube, proposal_cubes):
    """
    Compute the Intersection over Union (IoU) of two 3D cubes.

    Parameters:
    - gt_cube: GT Cube.
    - proposal_cube: List of Proposal Cubes.

    Returns:
    - iou: Intersection over Union (IoU) value.
    """
    gt_corners = torch.stack([gt_cube.get_all_corners()])
    proposal_corners = torch.stack([cube.get_all_corners() for cube in proposal_cubes]).to(gt_corners.device)

    # TODO check if corners in correct order; Should be
    vol, iou = box3d_overlap(gt_corners,proposal_corners)
    iou = iou[0]

    return iou

def custom_mapping(x,beta=1.7):
    '''
    maps the input curve to be S shaped instead of linear
    
    Args:
    beta: number > 1, higher beta is more aggressive
    x: list of floats betweeen and including 0 and 1
    beta: number > 1 higher beta is more aggressive
    '''
    mapped_list = []
    for i in range(len(x)):
        if x[i] <= 0:
            mapped_list.append(0.0)
        else:
            mapped_list.append((1 / (1 + (x[i] / (1 - x[i])) ** (-beta))))
    
    return mapped_list

def mask_iou(segmentation_mask, bube_mask):
    '''
    Area is of segmentation_mask
    '''
    # Compute intersection mask
    intersection_mask = np.logical_and(segmentation_mask, bube_mask).astype(np.uint8)
    # Count pixels in intersection
    intersection_area = np.sum(intersection_mask)

    # Compute union mask
    union_mask = np.logical_or(segmentation_mask, bube_mask).astype(np.uint8)
    # Count pixels in union mask
    union_area = np.sum(union_mask)

    # Compute IoU
    iou = intersection_area / union_area

    return iou

def is_gt_included(gt_cube,x_range,y_range,z_range, w_prior, h_prior, l_prior):
    # Define how far away dimensions need to be to be counted as unachievable
    stds_away = 1.5
    # Center
    because_of = []
    if not (x_range[0] < gt_cube.center[0] < x_range[1]):
        if (gt_cube.center[0] < x_range[0]):
            val = abs(x_range[0] - gt_cube.center[0])
        else:
            val = abs(gt_cube.center[0] - x_range[1])
        because_of.append(f'x by {val:.1f}')
    if not (y_range[0] < gt_cube.center[1] < y_range[1]):
        if (gt_cube.center[1] < y_range[0]):
            val = abs(y_range[0] - gt_cube.center[1])
        else:
            val = abs(gt_cube.center[1] - y_range[1])
        because_of.append(f'y by {val:.1f}')
    # Depth
    if not (z_range[0] < gt_cube.center[2] < z_range[1]):
        if (gt_cube.center[2] < z_range[0]):
            val = abs(z_range[0] - gt_cube.center[2])
        else:
            val = abs(gt_cube.center[2] - z_range[1])
        because_of.append(f'z by {val:.1f}')
    # Dimensions
    if (gt_cube.dimensions[0] < w_prior[0]-stds_away*w_prior[1]):
        because_of.append('w-')
    if (gt_cube.dimensions[0] > w_prior[0]+stds_away*w_prior[1]):
        because_of.append('w+')
    if (gt_cube.dimensions[1] < h_prior[0]-stds_away*h_prior[1]):
        because_of.append('h-')
    if (gt_cube.dimensions[1] > h_prior[0]+stds_away*h_prior[1]):
        because_of.append('h+')
    if (gt_cube.dimensions[2] < l_prior[0]-stds_away*l_prior[1]):
        because_of.append('l-')
    if (gt_cube.dimensions[2] > l_prior[0]+stds_away*l_prior[1]):
        because_of.append('l+')
    if because_of == []:
        return True
    else:
        print('GT cannot be found due to',because_of)
        return False

    # rotation nothing yet

def euler_to_unit_vector(eulers):
    """
    Convert Euler angles to a unit vector.
    """
    yaw, pitch, roll = eulers
    
    # Calculate the components of the unit vector
    x = np.cos(yaw) * np.cos(pitch)
    y = np.sin(yaw) * np.cos(pitch)
    z = np.sin(pitch)
    
    # Normalize the vector
    length = np.sqrt(x**2 + y**2 + z**2)
    unit_vector = np.array([x, y, z]) / length
    
    return unit_vector


# helper functions for plotting segmentation masks
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_mask2(masks:np.array, im:np.array, random_color=False):
    """
    Display the masks on top of the image.

    Args:
        masks (np.array): Array of masks with shape (h, w, 4).
        im (np.array): Image with shape (h, w, 3).
        random_color (bool, optional): Whether to use random colors for the masks. Defaults to False.

    Returns:
        np.array: Image with masks displayed on top.
    """
    im_expanded = np.concatenate((im, np.ones((im.shape[0],im.shape[1],1))*255), axis=-1)/255

    mask_image = np.zeros((im.shape[0],im.shape[1],4))
    for i, mask in enumerate(masks):
        if isinstance(random_color, list):
            color = random_color[i]
        else:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        h, w = mask.shape[-2:]
        mask_sub = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = mask_image + mask_sub
    mask_binary = (mask_image > 0).astype(bool)
    im_out = im_expanded * ~mask_binary + (0.5* mask_image + 0.5 * (im_expanded * mask_binary))
    im_out = im_out.clip(0,1)
    return im_out
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
