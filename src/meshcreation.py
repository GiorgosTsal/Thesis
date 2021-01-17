# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 19:18:32 2021

@author: root
"""

import numpy as np
import open3d as o3d
import os


input_path="datasets/"
output_path="output/"
# dataname="sample.xyz"

dataname="functionA_4.xyz" 
# datanamebunny = "bunny.ply"

# bmesh = o3d.io.read_triangle_mesh(input_path + datanamebunny)
# bmesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([bmesh])
# pcd = bmesh.sample_points_uniformly(number_of_points=500)
# o3d.visualization.draw_geometries([pcd])

# fun_path = input_path + dataname
fun_output_path = output_path+"funcA.ply"

if not os.path.exists(output_path):
    os.makedirs(output_path)
    

# point_cloud= np.loadtxt(input_path+dataname,skiprows=1)
point_cloud= np.loadtxt(input_path+dataname)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])


o3d.visualization.draw_geometries([pcd],width=1200, height=1000)

distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))

o3d.visualization.draw_geometries([pcd, mesh],width=1200, height=1000)


print()
print('Mesh Vertices:')
print(np.asarray(mesh.vertices))
print('Mesh Triangles:')
print(np.asarray(mesh.triangles))
print("Computing normal and rendering it.")
mesh.compute_vertex_normals()
print(np.asarray(mesh.triangle_normals))
print('================================================')




# Store mesh into .ply file
o3d.io.write_triangle_mesh(output_path+"sample.ply", mesh, True)


# # With Ball Pivoting Algorithm

# #http://www.open3d.org/html/tutorial/Advanced/surface_reconstruction.html
# # =======================Start=========================

# radii = [0.005, 0.01, 0.02, 0.04]
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#                pcd, o3d.utility.DoubleVector(radii))


# o3d.visualization.draw_geometries([pcd, rec_mesh])

# # Export and visualize
# # with the write_triangle_mesh function. We just specify within the name of the 
# # created file, the extension that we want from .ply, .obj, .stl or .gltf, 
# # and the mesh to export the BPA reconstructions as .ply files

# o3d.io.write_triangle_mesh(fun_output_path, rec_mesh, True) #write_ascii = True

# # Generate Levels of Details (LoD)
# # The function will take as parameters a mesh, a list of LoD (as a target number
# # of triangles), the file format of the resulting files and the path to write 
# # the files to

# def lod_mesh_export(mesh, lods, extension, path):
#     mesh_lods={}
#     for i in lods:
#         mesh_lod = mesh.simplify_quadric_decimation(i)
#         o3d.io.write_triangle_mesh(path+"lod_"+str(i)+extension, mesh_lod, True)
#         mesh_lods[i]=mesh_lod
#     print("generation of "+str(i)+" LoD successful")
#     return mesh_lods


# my_lods = lod_mesh_export(rec_mesh, [100000,50000,10000,1000,100], ".ply", output_path)

# # access and visualize 100 triangles
# o3d.visualization.draw_geometries([my_lods[100]])

##======================End==========================








# distances = pcd.compute_nearest_neighbor_distance()
# avg_dist = np.mean(distances)
# radius = 3 * avg_dist

# bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))

# o3d.io.write_triangle_mesh(output_path+"bpa_mesh.ply", bpa_mesh, True)

# def lod_mesh_export(mesh, lods, extension, path):
#     mesh_lods={}
#     for i in lods:
#         mesh_lod = mesh.simplify_quadric_decimation(i)
#         o3d.io.write_triangle_mesh(path+"function_lod_"+str(i)+extension, mesh_lod,True)
#         mesh_lods[i]=mesh_lod
#     print("generation of "+str(i)+" LoD successful")
#     return mesh_lods


# my_lods = lod_mesh_export(bpa_mesh, [100000,50000,10000,1000,100], ".ply", output_path)

# o3d.visualization.draw_geometries([my_lods[100]])


print('Program terminated succesfully...')