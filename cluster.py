
import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

def segment_clusters(pcd, distance_threshold=2, min_samples=30):
    """
    Segment clusters in a point cloud using DBSCAN clustering.
    """
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=distance_threshold, min_points=80, print_progress=True))

    max_label = labels.max()
    clusters = []
    ind=[]
    colors = o3d.utility.Vector3dVector(np.random.rand(max_label + 1, 3))  # Generate random colors
    for i in range(max_label + 1):
        component = pcd.select_by_index(np.where(labels == i)[0])
        
        if component.has_points():
            ind.append(np.where(labels == i)[0])
            component.paint_uniform_color(colors[i])
            clusters.append(component)

    return clusters,ind


def compute_bounding_boxes(buildings):
    """
    Compute bounding boxes for a list of buildings.
    """
    return [bldg.get_axis_aligned_bounding_box() for bldg in buildings]

def check_overlap(bbox1, bbox2):
    """
    Check if two bounding boxes overlap.
    """
    min_bound_1 = np.asarray(bbox1.min_bound)
    max_bound_1 = np.asarray(bbox1.max_bound)
    min_bound_2 = np.asarray(bbox2.min_bound)
    max_bound_2 = np.asarray(bbox2.max_bound)

    # Check for overlap in each dimension
    overlap_x = (min_bound_1[0] < max_bound_2[0]) and (max_bound_1[0] > min_bound_2[0])
    overlap_y = (min_bound_1[1] < max_bound_2[1]) and (max_bound_1[1] > min_bound_2[1])
    overlap_z = (min_bound_1[2] < max_bound_2[2]) and (max_bound_1[2] > min_bound_2[2])

    return overlap_x and overlap_y and overlap_z

'''

#test
if __name__ == '__main__':
        # Load point clouds (assuming pcd1 and pcd2 are in .ply format here)
    ucloud = o3d.io.read_point_cloud("m0.ply")
    mcloud = o3d.io.read_point_cloud("u0.ply")

    # Segment buildings in each point cloud
    clu1,ind1 = segment_clusters(ucloud)
    clu2,ind2 = segment_clusters(mcloud)

    # o3d.visualization.draw_geometries(clu1 )
    # o3d.visualization.draw_geometries(clu2 )

    # Compute bounding boxes for each building
    bboxes1 = compute_bounding_boxes(clu1)
    bboxes2 = compute_bounding_boxes(clu2)

    # Check for overlaps and delete overlapping buildings
    to_delete_from_pcd1 = set()
    to_delete_from_pcd2 = set()

    for i, bbox1 in enumerate(bboxes1):
        for j, bbox2 in enumerate(bboxes2):
            if check_overlap(bbox1, bbox2):
                to_delete_from_pcd1.add(i)
                to_delete_from_pcd2.add(j)


    # Convert sets to sorted lists in descending order
    to_delete_from_pcd1 = sorted(list(to_delete_from_pcd1), reverse=True)
    to_delete_from_pcd2 = sorted(list(to_delete_from_pcd2), reverse=True)


    # Remove overlapping buildings from UAV
    for i in sorted(to_delete_from_pcd1, reverse=True):
        del clu1[i]
        del ind1[i]

    # for j in sorted(to_delete_from_pcd2, reverse=True):
    #     del buildings2[j]

    # Visualize the remaining buildings
    # o3d.visualization.draw_geometries(clu1 + clu2)

    clu2.append(mcloud)
    clu1.append(ucloud)

    o3d.visualization.draw_geometries(clu1 + clu2)'''