import open3d as o3d
import numpy as np

from collections import defaultdict
import os
import re
import laspy as las
import datetime
from scipy.spatial import KDTree
import copy



def weited_guided_filter(source,target,radius, epsilon, w1=0.8):
    """
    guided filter of source weited by the similarity of source and target
    
    """
     # # Compute normals
    source.estimate_normals()
    target.estimate_normals()
    src_nm=source.normals
    tgt_nm=target.normals

    pt_src = np.array(source.points)
    pt_tgt = np.array(target.points)

    # 1. Build KD-Tree for both pcd1 and pcd2
    tree_src = KDTree(np.asarray(pt_src))
    tree_tgt = KDTree(np.asarray(pt_tgt))
    #w1=0.8
    w2=1-w1
    
    #find corresbonding in uav of every mm points
    for i, point in enumerate(tqdm(pt_src)):
        
        index_tgt = tree_tgt.query_ball_point(point,r=radius,return_sorted=False)
        index_self = tree_src.query_ball_point(point,r=1/2*radius,return_sorted=False)
        
        if len(set(index_tgt))>8 and len(set(index_self))>5 :
            nnin=index_tgt[0]

            d=np.linalg.norm(point - pt_tgt[nnin])
            
            src_nm_n= src_nm[i]/src_nm[i].dot( src_nm[i])**0.5             
            tgt_nm_n= tgt_nm[nnin]/tgt_nm[nnin].dot(tgt_nm[nnin])**0.5 

            cos_a=abs(np.dot(src_nm_n,tgt_nm_n))

            num=len(index_tgt)
            num_tgt=int(num*(w1*cos_a+w2*(1-d/0.8*radius)))
            num_self=max(num-num_tgt,8)

            neighbors=[]

            for ind in index_tgt[0: num_tgt]:
                neighbors.append(pt_tgt[ind])

            for ind in index_self[0:num_self]:
                neighbors.append(pt_src[ind])   

            neighbors=np.asarray(neighbors)
            mean = np.mean(neighbors,axis=0)
            # local covariance
            cov = np.cov(neighbors.T)
            e = np.linalg.inv(cov + epsilon * np.eye(3))
            
            A = cov @ e
            b = mean - A @ mean        
            pt_src[i] = A @  pt_src[i] + b
    
    pc=o3d.geometry.PointCloud()

    pc.points = o3d.utility.Vector3dVector(pt_src)
    pc.colors = o3d.utility.Vector3dVector(source.colors)

    return pc



def compute_local_curvature(neighbors):
    """
    Compute local curvature using PCA on the neighborhood.
    """
    # mean = np.mean(neighbors, axis=0)
    cov_matrix = np.cov(neighbors, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)
    smallest_eigenvalue = eigenvalues[sorted_indices[0]]

    return smallest_eigenvalue / (eigenvalues.sum())




def edge_aware_guided_filter(source, target,radius, epsilon):
    """
    guided filter of pointclouds
    """
    # Compute normals
    source.estimate_normals()
    target.estimate_normals()

    pt_src = np.array(source.points)
    pt_tgt = np.array(target.points)

    # 1. Build KD-Tree for both pcd1 and pcd2
    tree_tgt = KDTree(np.asarray(pt_tgt))
    edge_indices=[]
    #find corresbonding in uav of every mm points
    for i, point in enumerate(pt_src):
        print(i)
        index_tgt = tree_tgt.query_ball_point(point,r=radius,return_sorted=False)
        if len(index_tgt)>10 :
            neighbors=[]
            for ind in index_tgt:
                neighbors.append(pt_tgt[ind]) 

            local_curvature = compute_local_curvature(neighbors)
               
            if local_curvature>0.04:
                edge_indices.append(i)
            else:
                #low curvature more neibors
                num=int((1-local_curvature*local_curvature/0.016)*len(index_tgt))
                if num>3:                                
                    neighbors=np.asarray(neighbors[:num])              
                    mean = np.mean(neighbors,axis=0)        
                    # local covariance
                    cov = np.cov(neighbors.T)
                    e = np.linalg.inv(cov + epsilon * np.eye(3))
                    
                    A = cov @ e
                    b = mean - A @ mean
                    pt_src[i] = A @  pt_src[i] + b
    
    edge=source.select_by_index(edge_indices)
    o3d.visualization.draw_geometries([edge])

    pc=o3d.geometry.PointCloud()

    pc.points = o3d.utility.Vector3dVector(pt_src)
    pc.colors = o3d.utility.Vector3dVector(source.colors)

    return pc

def progressive_filter(source, target,radius):
    """
    progressively move the source point to target point  
    """
    # Compute normals
    source.estimate_normals()
    target.estimate_normals()

    src_normal = np.asarray(source.normals)


    src_pt = np.array(source.points)
    tg_pt = np.array(target.points)

    # 1. Build KD-Tree for both pcd1 and pcd2
    src_tree = KDTree(np.asarray(src_pt))
    tg_tree = KDTree(np.asarray(tg_pt))
    d=[]
    #find corresbonding in uav of every mm points
    for i, point in enumerate(src_pt):
        print(i)
        #index = treeu.query_ball_point(point,r=radius,return_sorted=False)
        d_src2tg,nn_src2tg = tg_tree.query(point)
        d_tg2src,nn_tg2src = src_tree.query(tg_pt[nn_src2tg])

        bidis=(d_tg2src+d_src2tg)/2
        d.append(bidis)

        if bidis<0.3:

            norm_normalu=src_normal[i]/(src_normal[i].dot(src_normal[i])**0.5 )
            # Compute vector 
            vect = np.array(tg_pt[nn_src2tg]) - np.array(point)
            norm_vect=vect/(vect.dot(vect)**0.5 )

            proj_dis= abs(np.dot(vect,  norm_normalu))

            T=0.1

            D=proj_dis*norm_vect

            if proj_dis<T:
                src_pt[i] =src_pt[i]+ D
    
    pc=o3d.geometry.PointCloud()

    pc.points = o3d.utility.Vector3dVector(src_pt)

    return pc

import math
from tqdm import tqdm
'''
def filter_points_by_distance(sourcepcd, targetpcd,use_normal_dis, distance_threshold=0.05 ):
    """
    Returns points in source where the corresponding distance exceeds the distance threshold 
    and the angle between normals exceeds the angle threshold.
    """

    filtered_points = []

    cl,ind=targetpcd.remove_radius_outlier( 50, 0.2, print_progress=True)
    targetpcd=targetpcd.select_by_index(ind)
    o3d.visualization.draw_geometries([  targetpcd],window_name='filteru')


   # sourcepcd.estimate_normals( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=100))
    targetpcd.estimate_normals( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=90))

    source=sourcepcd.points
    target=targetpcd.points

    # src_normals= np.asarray(sourcepcd.normals)
    tgt_normals= np.asarray(targetpcd.normals)
    #Build KD-Tree
    tgt_tree = KDTree(np.asarray(target))
    mettt=np.zeros(len(source))
    for i, point in enumerate(tqdm(source)):

        dis, index = tgt_tree.query(point)


        #find the project distance on the direction of normal of target point
        nn = target[index]
        norm_srnormal= tgt_normals[i]/tgt_normals[i].dot( tgt_normals[i])**0.5 
        vect = np.array(nn) - np.array(point)

        proj= (np.dot(vect, norm_srnormal))
        plandis=math.sqrt(np.linalg.norm(vect)**2-proj**2)

        if use_normal_dis:
            met=1
        else:
            met=math.exp(-1*plandis*plandis/distance_threshold*distance_threshold)
          
        if met<255/256:
            filtered_points.append(i)

    pc=sourcepcd.select_by_index(filtered_points)

    o3d.visualization.draw_geometries([pc])


    return pc
'''

def filter_points_by_plane_distance(sourcepcd, targetpcd, distance_threshold=0.1, use_normal_dis=False):
    """
    Returns points in source where the corresponding distance exceeds the distance threshold 
    and the angle between normals exceeds the angle threshold.
    """

    filtered_points = []

    cl,ind=targetpcd.remove_radius_outlier( 50, 0.2, print_progress=True)
    targetpcd=targetpcd.select_by_index(ind)

    targetpcd.estimate_normals( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=90))
    
    source=sourcepcd.points
    target=targetpcd.points

    tgt_normals= np.asarray(targetpcd.normals)

    #Build KD-Tree
    tgt_tree = KDTree(np.asarray(target))
    
    for i, point in enumerate(tqdm(source)):

        dis, index = tgt_tree.query(point)


        #find the project distance on the plane
        nn = target[index]
        norm_srnormal= tgt_normals[i]/tgt_normals[i].dot( tgt_normals[i])**0.5 
        vect = np.array(nn) - np.array(point)
        proj= (np.dot(vect, norm_srnormal))
        plandis=math.sqrt(np.linalg.norm(vect)**2-proj**2)
        
        if use_normal_dis:
            met=math.exp(-1*plandis*plandis/distance_threshold*distance_threshold)* math.exp(proj*proj/distance_threshold*distance_threshold)
        
        else:
            #met=math.exp(-1*plandis*plandis/distance_threshold*distance_threshold)
            met=math.exp(-1*plandis*plandis/distance_threshold*distance_threshold)
  
        if met<255/256:
            filtered_points.append(i)

    pc=sourcepcd.select_by_index(filtered_points)

    return pc

