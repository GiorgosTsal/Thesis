# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 22:19:22 2020

@author: root
"""

import os
#himport numpy as np
import open3d as o3d
#import matplotlib.pyplot as plt



input_path="datasets/"
output_path="output/"
dataname="bunny.ply"

bunny_path = input_path + dataname

if not os.path.exists(output_path):
    os.makedirs(output_path)
    
bunny_output_path = output_path+"bpa_mesh.ply"

mesh = o3d.io.read_triangle_mesh(bunny_path)
mesh.compute_vertex_normals()
pcd = mesh.sample_points_poisson_disk(3000)


o3d.visualization.draw_geometries([pcd])

# With Ball Pivoting Algorithm

#http://www.open3d.org/html/tutorial/Advanced/surface_reconstruction.html
# =======================Start=========================

radii = [0.005, 0.01, 0.02, 0.04]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
               pcd, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([pcd, rec_mesh])

# Export and visualize
# with the write_triangle_mesh function. We just specify within the name of the 
# created file, the extension that we want from .ply, .obj, .stl or .gltf, 
# and the mesh to export the BPA reconstructions as .ply files

o3d.io.write_triangle_mesh(bunny_output_path, rec_mesh, True) #write_ascii = True

# Generate Levels of Details (LoD)
# The function will take as parameters a mesh, a list of LoD (as a target number
# of triangles), the file format of the resulting files and the path to write 
# the files to

def lod_mesh_export(mesh, lods, extension, path):
    mesh_lods={}
    for i in lods:
        mesh_lod = mesh.simplify_quadric_decimation(i)
        o3d.io.write_triangle_mesh(path+"lod_"+str(i)+extension, mesh_lod, True)
        mesh_lods[i]=mesh_lod
    print("generation of "+str(i)+" LoD successful")
    return mesh_lods


my_lods = lod_mesh_export(rec_mesh, [100000,50000,10000,1000,100], ".ply", output_path)

# access and visualize 100 triangles
o3d.visualization.draw_geometries([my_lods[100]])

##======================End==========================



print("Program terminated successfully...")