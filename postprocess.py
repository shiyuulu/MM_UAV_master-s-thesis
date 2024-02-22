import open3d as o3d
import numpy as np


import os

import laspy as las
import datetime

from cluster import*
from guided_filter import*

'''
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
'''

#find lastest generated folder
def get_latest_folder_with_timestamp(parent_folder):

    # List all folders in the parent folder
    subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]

    # Extract timestamps and corresponding folder names
    timestamp_folder_map = {}
    for subfolder in subfolders:
        try:
            timestamp = datetime.datetime.strptime(subfolder, "%Y-%m-%d_%H-%M-%S")
            timestamp_folder_map[timestamp] = subfolder
        except ValueError:
            # Skip folders that don't match the timestamp format
            pass

    # Find and return the folder with the latest timestamp
    if timestamp_folder_map:
        latest_timestamp = max(timestamp_folder_map.keys())
        latest_folder = timestamp_folder_map[latest_timestamp]
        return os.path.join(parent_folder, latest_folder)
    else:
        return None

def getpcd(las,offset=[0,0,0]):
    '''
    generate open3d point cloud from laspy
    '''
    las.points.offsets=offset
    points = np.vstack((las.x, las.y, las.z)).transpose()
    colors = np.vstack((las.red, las.green, las.blue)).transpose()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 65535)
    return pcd

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

def remove_overlap(cloudu,cloudm,eps=2,min_ptmum=60):
    '''
    remove the overlap clusters in cloud1 according to bbx
    '''

     # Segment clusters in each point cloud

    cloudm.paint_uniform_color([0,0,1])
    cloudu.paint_uniform_color([1,0,0])
    candidateu,indu = segment_clusters(cloudu,eps,min_ptmum)
    candidatem,indm = segment_clusters(cloudm,eps,min_ptmum)

    o3d.visualization.draw_geometries(candidateu)
    o3d.visualization.draw_geometries(candidatem)

    # Compute bounding boxes for each building
    bboxesu = compute_bounding_boxes(candidateu)
    bboxesm = compute_bounding_boxes(candidatem)

    # Check for overlaps and delete overlapping buildings
    to_delete_from_pcdu = set()
    to_delete_from_pcdm = set()

    for i, bbox1 in enumerate(bboxesu):
        for j, bbox2 in enumerate(bboxesm):
            if check_overlap(bbox1, bbox2):
                to_delete_from_pcdu.add(i)
                to_delete_from_pcdm.add(j)


    # Convert sets to sorted lists in descending order
    to_delete_from_pcdu = sorted(list(to_delete_from_pcdu), reverse=True)
    to_delete_from_pcdm = sorted(list(to_delete_from_pcdm), reverse=True)


    # Remove overlapping buildings
    for i in sorted(to_delete_from_pcdu, reverse=True):
        del candidateu[i]
        del indu[i]


    cloudm_selected=cloudm.select_by_index( np.hstack(indm))
    
    if len(indu):
        cloudu_selected=cloudu.select_by_index( np.hstack(indu))
        merged=merge_point_clouds(cloudu_selected,cloudm_selected)

    else:
        merged= cloudm_selected

    o3d.io.write_point_cloud('after.ply',cloudm_selected)
    o3d.io.write_point_cloud('afteruu.ply',cloudu_selected)
    

    
    o3d.visualization.draw_geometries([merged])

    candidateu.append(cloudu)
    candidateu.append(cloudm)

    # o3d.visualization.draw_geometries(candidateu + candidatem)


    o3d.visualization.draw_geometries([cloudm,cloudu])
    o3d.io.write_point_cloud('mmunk.ply',cloudm)
    o3d.io.write_point_cloud('uav_unk.ply',cloudu)

    return merged



# fusion for constructed surface
def constructed_surface(cloudu,cloudm, voxel_size=0.5, radius=0.3, epsilon=0.01, use_normal_dis=False):
    
    candidateu=filter_points_by_plane_distance(cloudu, cloudm, distance_threshold=0.1,use_normal_dis=use_normal_dis)

    o3d.io.write_point_cloud('candidateu.ply',candidateu)

    filtered_u=weited_guided_filter(candidateu, cloudm, radius, epsilon)
    filtered_u=guided_filter(candidateu, cloudm, radius, epsilon)

    candidateu.paint_uniform_color([1,0,0])
    o3d.visualization.draw_geometries([filtered_u, candidateu])

    # o3d.io.write_point_cloud('filtered_u.ply',filtered_u)
    merged=merge_point_clouds(filtered_u,cloudm)

    o3d.visualization.draw_geometries([merged])

    return merged
    


#number in LAS files 
file_name=[
           '8222_106327',
           '8212_106333',           
           '8237_106335',
           '8258_106297',
           '8306_106370']

#root path of LAS files
path="/home/lu/Test/Thesis/Data/new/"


#name of registration results
pfn=['mm.ply',
     'uav.ply',
     'uav_alined.ply',
     'uav_alined_ICP.ply',
     'uav_stable_alined.ply',
     'uav_stable_alined_ICP.ply'
     ]


def main():
    # Regular expression to match the desired format
    for na in list(file_name):
        
        rootf=os.path.join(path,na)
        latest_folder = get_latest_folder_with_timestamp(rootf)
        
        
        transpath=os.path.join(latest_folder,'trans.txt')
        transdata= np.loadtxt(transpath,skiprows=0)

        # List all files in the specified directory
        for f in os.listdir(path):
            if na in f and '.las' in f:
                lasfiles=os.path.join(path, f)

        laz = las.read(lasfiles)

        lazu=laz[laz.points['Original cloud index']==0]
        lazm=laz[laz.points['Original cloud index']==1]

        #remember the global offset 
        offset=lazm.points.offsets

        orim=getpcd(lazm)
        oriu=getpcd(lazu)


        oriu_vis=copy.deepcopy(oriu)

        oriu_vis.paint_uniform_color([1,0,0])
        o3d.visualization.draw_geometries([oriu_vis,orim])

        #transformation matrix of random, feature based, ICP
        trans=[transdata[:4,:],transdata[4:8,:],transdata[8:12,:]]

        uav_alined=oriu.transform(trans[0]).transform(trans[1]).transform(trans[2])

        #o3d.io.write_point_cloud('uaalin.ply',uu)

        constantu=np.asarray(lazu.points['Constant'])
        constantm=np.asarray(lazm.points['Constant'])

        classes={
        'unknow':[0],
        'building':[1,2],
        'surface':[9,10,11],
        'vegetation':[3,4,5,6,7,8],
        'car':[12],
        'dynamic':[13]
        }


        pcd=[]

        for clas_na,ind in classes.items():
            
            u_ind=list(np.where(np.in1d(constantu,ind))[0])
            m_ind=list(np.where(np.in1d(constantm,ind))[0])

            u_sub=uav_alined.select_by_index(u_ind)
            m_sub=orim.select_by_index(m_ind)

            u_vis=copy.deepcopy(u_sub)
            u_vis.paint_uniform_color([1,0,0])
            
            o3d.visualization.draw_geometries([u_vis,m_sub])
            
            save_folder = os.path.join(path, na,'post_sub')
            os.makedirs(save_folder, exist_ok=True)


            #strategy for small object
            if clas_na=='unknow':
                #continue
                print("start processing of",format(clas_na))
                if len(u_sub.points) and len(m_sub.points):
                    merged=remove_overlap(u_sub,m_sub,eps=1.2,min_ptmum=30)
                    pcd.append(merged)
                    # o3d.io.write_point_cloud(os.path.join(save_folder,str(na)+'unkonw.ply'),merged)
                else:
                    print('No unknow class in the scene')
                    continue

            #strategy for small object
            elif clas_na== 'car':
                # continue
                print("start processing of",format(clas_na))
                if len(u_sub.points) and len(m_sub.points):
                    merged=remove_overlap(u_sub,m_sub,eps=0.5,min_ptmum=80)
                    pcd.append(merged)
                    # o3d.io.write_point_cloud(os.path.join(save_folder,str(na)+'car.ply'),merged)
                else:
                    print('No car in the scene')
                    continue

            #strategy for building and ground surface
            elif clas_na=='building':
                #continue
                o3d.io.write_point_cloud(os.path.join(save_folder,str(na)+'bu.ply'),u_sub)
                o3d.io.write_point_cloud(os.path.join(save_folder,str(na)+'bm.ply'),m_sub)

                print("start processing of",format(clas_na))
                merged= constructed_surface(u_sub,m_sub,  radius=0.3, epsilon=0.01)
                pcd.append(merged)
                # o3d.io.write_point_cloud(os.path.join(save_folder,str(na)+'building.ply'),merged)
         
            #strategy for building and ground surface
            elif clas_na=='surface':
                #continue
                o3d.io.write_point_cloud(os.path.join(save_folder,str(na)+'su.ply'),u_sub)
                o3d.io.write_point_cloud(os.path.join(save_folder,str(na)+'sm.ply'),m_sub)
                print("start processing of",format(clas_na))
                merged= constructed_surface(u_sub,m_sub,  radius=0.3, epsilon=0.01)
                pcd.append(merged)
                # o3d.io.write_point_cloud(os.path.join(save_folder,str(na)+'surface.ply'),merged)

            #strategy for vegetation
            elif clas_na=='vegetation':
                #continue
                print("start processing of",format(clas_na))
                if len(u_sub.points) and len(m_sub.points):
                    merged=merge_point_clouds(u_sub,m_sub)
                    pcd.append(merged)
                    cl,ind= merged.remove_statistical_outlier(nb_neighbors=30,std_ratio=4.0)
                    merged= merged.select_by_index(ind)
                    o3d.io.write_point_cloud(os.path.join(save_folder,str(na)+'vegetation.ply'),merged)
                else:
                    print('No car in the scene')
                    continue
            
            #remove dynamic
            elif clas_na=='dynamic':
                # print("start processing of {}:",format(clas_na))
                continue
        
        mergefinal= pcd[0]
       
        for pt in pcd[1:]:
            
            mergefinal=merge_point_clouds(pt,mergefinal)


        o3d.visualization.draw_geometries([mergefinal])

        o3d.io.write_point_cloud(os.path.join(save_folder,str(na)+'finalpt.ply'),mergefinal)




if __name__ == "__main__":

    main()


