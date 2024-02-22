import numpy as np 
import scipy.sparse
from collections import OrderedDict
import open3d as o3d

from tqdm import tqdm
from scipy.sparse import csr_matrix

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize


from scipy.sparse.linalg import inv

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

def sample_points(points, scores, n_samples, method):
    
    def sample_points_topk(pcd, scores, n_samples):
        """ sample points with top K scores.

        Args:
            points: np.ndarray 
                shape (N, 3)

            scores: np.ndarray
                score for each point . shape (N,)

            n_samples: int
                number of sampled points

        Returns:
            points_sampled: np.ndarray
                shape (n_samples, 3)
            
        """
    
        # Find indices where scores are not 1
        points = np.asarray(pcd.points)
        scores=np.asarray(scores)
        valid_indices = np.where(scores != 1)[0]

        # Sort valid indices by scores in descending order and select top n_samples
        top_k_indices = np.argsort(-scores[valid_indices])[:n_samples]

        # Select the corresponding points
        points_sampled = pcd.select_by_index(valid_indices[top_k_indices])
        # points[valid_indices[top_k_indices]]


        return points_sampled

    def sample_points_prob(points, scores, n_samples):
        """ sample points according to score normlized probablily

        Args:
            points: np.ndarray 
                shape (N, 3)

            scores: np.ndarray
                score for each point . shape (N,)

            n_samples: int
                number of sampled points

        Returns:
            points_sampled: np.ndarray
                shape (n_samples, 3)
            
        """
        scores=np.asarray(scores)
        valid_indices = np.where(scores != 1)[0]
        scores=scores[valid_indices]
        scores = scores / np.sum(scores)
        ids_sampled = np.random.choice(
            len(valid_indices), n_samples, replace=False, p=scores)
        points_sampled = points[ids_sampled]
        return points_sampled
    if method == 'prob':
        return sample_points_prob(points, scores, n_samples)
    elif method == 'topk':
        return sample_points_topk(points, scores, n_samples)



def compute_adjacency_matrix(kd_tree, pcd_np,pcd_cov, radius, max_neighbor, var=None):
    """ compute adjacency matrix 
    Notice: for very sparse point cloud, you may not find any neighbor within radius,
    Then the row for this point in matrix is all zero. Will cause bug. 
    Maybe replace search_hybrid_vector_3d with search_knn_vector_3d
    

    Args:
        kdtree: o3d.geometry.KDTreeFlann(pcd)
            kdtree for point cloud

        pcd_np: np.ndarray
            shape (N, 3)

        radius: float
            searching radius for nearest neighbor

        max_neighbor: int
            max number of nearest neighbor

        var: float


    Returns:
        adj_matrix_new: scipy.sparse.coo_matrix

    """
    N = pcd_np.shape[0]
    #adj_dict = OrderedDict()
    adj_matrix = scipy.sparse.dok_matrix((N, N))
    
   
    # points = np.asarray(pcd_np.points)
   # distances = np.zeros(N)
    dis=[]
    
    for i in tqdm(range(N)):
        [k, idx, dist_square] = kd_tree.search_hybrid_vector_3d(pcd_np[i], radius, max_neighbor)
        
        #curv=compute_local_curvature(pcd_cov[i])
    
        dist_square_value = np.asarray(dist_square)[0]
        # * np.exp(-(curv*curv)/0.0004)
        #dis.append(curv)

        adj_matrix[i, idx] = np.exp(-dist_square_value / radius**2) 

     
    adj_matrix = adj_matrix.tocoo()
    row, col, dist_square = adj_matrix.row, adj_matrix.col, adj_matrix.data

    data_new = np.exp(- dist_square )
    adj_matrix_new = scipy.sparse.coo_matrix((data_new, (row, col)), shape=(N,N))

    return adj_matrix_new, dis



def compute_D(W):
    """ compute degree matrix
    Args:
        W: scipy.sparse.coo_matrix

    Returns:
        D: sparse.matrix
    """
    N = W.shape[0]

    diag = np.array(W.sum(axis=1)).flatten()
    D = scipy.sparse.coo_matrix((N, N))
    D.setdiag(diag)

    return D


def laplacian_smoothing(pcd_np, L, NF, F, gamma=0.1, f_gamma=0.9, num_iter=800):
    """
    Apply Laplacian smoothing to a point cloud.

    Args:
    pcd_np (np.ndarray): The original point cloud, shape (N, 3).
    L (np.ndarray): The graph Laplacian, shape (N, N).
    alpha (float): Regularization parameter.
    num_iter (int): Number of iterations for the optimization.

    Returns:
    np.ndarray: The smoothed point cloud, shape (N, 3).
    """
    N = pcd_np.shape[0]

    
    
    def objective(x):
        x = x.reshape(N, 3)
        # Sx=S@x
        laplacian_term =  gamma *np.sum((x.T@L@ x)**2)
        
        # np.sum(L.dot(x)* x  )     #laplacan loss
        #np.sum((x - pcd_np)**2)  #distance loss
        # np.sum((x - L.dot(x))**2)  #feature loss
        #np.sum(x.T@L@ x)            # #total vari
        # np.sum((L.dot(x))**2)       #smooth
        regularization_term = (1-gamma)*(f_gamma*np.linalg.norm(F @ (x - pcd_np))**2+(1-f_gamma)*np.sum(NF @ (x - pcd_np)**2)+np.sum((x - pcd_np)**2))

        return laplacian_term + regularization_term

    def gradient(x):
        x = x.reshape(N, 3)
        # Sx=S@x
        laplacian_grad =  gamma * 2 *L.dot(x)       
        # 2 * L.dot(x)              #laplacan loss
        # (x - pcd_np)                 #distance loss
        # L.T.dot(L.dot(x) - x)     #feature loss
        # L.dot(x)                     #total vari
        # *L.T.dot(L.dot(x))            #smooth
        regularization_grad =(1-gamma) *2 *(f_gamma*F @ (x - pcd_np)+ (1-f_gamma)*NF @(x - pcd_np)+ (x - pcd_np))
        return (laplacian_grad + regularization_grad).flatten()

    # works well:
    # smooth/ total ari +distance loss regularization.  gamma= 0.1

    # Initial guess
    x0 = pcd_np.flatten()

    # Optimize
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', options={'maxiter': num_iter})
    
    
    #print(f"Number of calls to Simulator instance {res.num_calls}")
    if res.success:
        message = "Optimization successful."
    else:
        message = "Optimization reached the maximum number of iterations."


    
    return res.x.reshape(N, 3)
   


def apply_filter(pcd_np, F,dis):
    """ 
    Args:
        pcd_np: np.ndarray 
            shape (N, 3)

        F: sparse.coo_matrix (N, N)
            graph filter

    Returns:
        scores: np.ndarray
            shape (N,)
    """
    scores=F @  pcd_np
    scores = np.linalg.norm(scores, ord=2, axis=1)  # [N], L2 distance
    scores = scores ** 2  # [N], L2 distance square 

    scores=np.asarray(scores)  
    cur= np.array(dis)
    thre=np.median(cur)
    # cur_ind=np.where (cur > thre)[0]
    

    n_sam=int(0.1*len(pcd_np))

     # Sort scores in descending order and get indices
    sorted_indices = np.argsort(-scores)

    # # Select top indices where dis condition is met
    sam = sorted_indices[int(0.01*len(pcd_np)):n_sam+int(0.01*len(pcd_np))]

    # filtered_scores_pt = np.vstack([scores_pt[sampled],pcd_np[sam]]) 
    filtered_scores_pt = pcd_np[sam]

    # Initialize the binary array with zeros
    features = np.zeros(len(pcd_np), dtype=int)  # N should be the total number of points in your point cloud

    # Set the entries at the feature indices to 1
    features[sam] = 1

    return filtered_scores_pt, scores,features


def merge_point_clouds(*point_clouds):
    """
    Merge multiple Open3D point clouds.

    Parameters:
    - *point_clouds: Any number of Open3D point clouds.

    Returns:
    - Merged Open3D point cloud.
    """
    
    # Check if there's at least one point cloud
    if not point_clouds:
        raise ValueError("At least one point cloud must be provided.")
    
    # Initialize lists to store points and colors
    all_points = []
    all_colors = []
    
    for pcd in point_clouds:
        all_points.append(np.asarray(pcd.points))
        
        # If point cloud has colors, store them
        if pcd.has_colors():
            all_colors.append(np.asarray(pcd.colors))

    # Concatenate all points and colors
    merged_points = np.vstack(all_points)

    # Check if all point clouds had colors
    if len(all_colors) == len(point_clouds):
        merged_colors = np.vstack(all_colors)
    else:
        merged_colors = None

    # Create a new point cloud with merged points
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
    
    if merged_colors is not None:
        merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

    return merged_pcd



def Feature_points_detecting_and_denoising(pcd, kd_tree,filter_type,  max_neighbor, var, voxel_size = 0.02 ):
    """ 
    Find feature points based on graph high pass filter
    Apply denoising based on laplacian smoothing 
    Perform simplification based on voxel downsimpling 
    """
  
    pcd_np = np.asarray(pcd.points)
    N = pcd_np.shape[0]
    
    #searching radius of point cloud graph representation 
    radius = 5*voxel_size

    pcd.estimate_covariances( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_neighbor))
    pcd_cov=pcd.covariances

    pcd.estimate_normals( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_neighbor))
    pcd_nm=np.asarray(pcd.normals)

    #build up adjacency matrix
    W,dis = compute_adjacency_matrix(kd_tree, pcd_np,pcd_cov, radius, max_neighbor, var)
    D = compute_D(W) 

      
    #High pass filiter
    F=D-W
    
    #apply high pass filter to find feature points 
    sample_pt,scores, feature = apply_filter(pcd_np, F,dis)

    #mask of feature points and non feature points
    NF_mask = diags(feature)
    F_mask=diags(feature)

    #apply denoising based on laplacian smoothing 
    scores_pt= laplacian_smoothing(pcd_np, F,NF_mask,F_mask, gamma=0.1,f_gamma=0.7)

    pcd_fea = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample_pt))

    pcd_af = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scores_pt))

   
    if len(pcd.colors):
        pcd_fea.colors=o3d.utility.Vector3dVector(np.array(pcd.colors)[feature==1])
        pcd_af.colors=o3d.utility.Vector3dVector(pcd.colors)

        
    # Apply voxel downsampling
    downsampled_pcd = pcd_af.voxel_down_sample(voxel_size)

    pcd_simplify=merge_point_clouds(pcd_fea.voxel_down_sample(2*voxel_size),downsampled_pcd)

    return pcd_simplify





def sample_pcd(pcd, filter_type,voxel_size,  max_neighbor, var, method = "prob"):
    """ 
    Generate sampled and denoised point cloud 

    Args:
        pcd: open3d.geometry.PointCloud

        var: float
            given variance. Set to None if use data's variance

    Returns:
        pcd_sampled: open3d.geometry.PointCloud
    """

    assert method in ['prob', 'topk']
    processed=[] 
    scores_p=[]
    kd_tree = o3d.geometry.KDTreeFlann(pcd)
    if len(pcd.points)>2000000:

        #batch processing 
        batch_size=2000000

        pcd_sort=o3d.geometry.PointCloud()
        ptlist=np.array(pcd.points)
        clist=np.array(pcd.colors)
        ptsort=ptlist[np.argsort(ptlist[:,2])]
        csort=clist[np.argsort(ptlist[:,2])]
        pcd_sort.points= o3d.utility.Vector3dVector(ptsort)
        pcd_sort.colors= o3d.utility.Vector3dVector(csort)
        o3d.visualization.draw_geometries([pcd_sort])

        

        for i in range(0, len(pcd_sort.points), batch_size):

            ind=np.arange(i,min(i+batch_size,len(pcd_sort.points)))

            batch_pcd= pcd_sort.select_by_index(ind)

            pcd_sim = Feature_points_detecting_and_denoising(batch_pcd, kd_tree, filter_type,  max_neighbor, var, voxel_size)

            processed.append(pcd_sim)

        pcd_simplify=merge_point_clouds(processed)
      

    else:
        pcd_simplify = Feature_points_detecting_and_denoising(pcd,  kd_tree,filter_type, max_neighbor, var, voxel_size)

        o3d.visualization.draw_geometries([pcd_simplify])

    return pcd_simplify






from scipy.spatial.transform import Rotation as RR

def rotation_matrix_to_angle(R):
    # Convert rotation matrix to axis-angle representation and return the angle
    r = RR.from_matrix(R)
    axis_angle = r.as_rotvec()
    angle = np.linalg.norm(axis_angle)  # Calculate 
    return angle
def calculate_errors(ground_truth_matrix, calculated_matrix):
    # Extract rotation and translation components
    R_gt = ground_truth_matrix[:3, :3]
    t_gt = ground_truth_matrix[:3, 3]
    R_calc = calculated_matrix[:3, :3]
    t_calc = calculated_matrix[:3, 3]

    # Calculate rotation error
    rotation_diff = np.dot(R_calc, np.linalg.inv(R_gt))
    rotation_error = rotation_matrix_to_angle(rotation_diff)

    # Calculate translation error
    translation_error = np.linalg.norm(t_calc - t_gt)

    return rotation_error, translation_error


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 1])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def fine_reg(source, target,feature_T, gt_T):
    '''
    fine registration

    source: source point cloud
    target: target point cloud 
    feature_T: coarse registration results
    gt_T:ground truth
    '''
  
    current_transformation = np.identity(4)


    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60))

    result_icp = o3d.pipelines.registration.registration_icp(
    source,target, 0.1, current_transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )


    # print(result_icp)

    
    draw_registration_result_original_color(source, target,
                                            result_icp.transformation)


    t_icp=np.dot(feature_T,result_icp.transformation)

 
    rotation_error, translation_error = calculate_errors(gt_T, t_icp)
    print(f"nosimplify_Rotation Error: {rotation_error} radians")
    print(f"Translation Error: {translation_error} units")


    return t_icp
