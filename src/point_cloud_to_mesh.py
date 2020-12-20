# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:38:18 2020

@author: Giorgos Tsalidis
"""


import numpy as np
import open3d as o3d

input_path="datasets/"
output_path="output/"
dataname="dragon.ply"

# Read .ply file
print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud(input_path+dataname)

# Visualize the point cloud within open3d
#o3d.visualization.draw_geometries([pcd])

# Convert open3d format to numpy array
# Here, you have the point cloud in numpy format. 
point_cloud_in_numpy = np.asarray(pcd.points) 

# Visualize the point cloud within open3d
#o3d.visualization.draw_geometries([pcd])

# meshing strategy
# Ball-Pivoting Algorithm

# first compute the necessary radius parameter based on the average distances
# computed from all the distances between points
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist

#create a mesh and store it in the bpa_mesh variable
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,
                                o3d.utility.DoubleVector([radius, radius * 2]))

# Before exporting the mesh, we can downsample the result to an acceptable 
# number of triangles, for example, 100k triangles
dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)
print(dec_mesh)
# run the following commands to ensure its consistency

dec_mesh.remove_degenerate_triangles()
dec_mesh.remove_duplicated_triangles()
dec_mesh.remove_duplicated_vertices()
dec_mesh.remove_non_manifold_edges()

# Export and visualize
# with the write_triangle_mesh function. We just specify within the name of the 
# created file, the extension that we want from .ply, .obj, .stl or .gltf, 
# and the mesh to export the BPA reconstructions as .ply files

o3d.io.write_triangle_mesh(output_path+"bpa_mesh.ply", dec_mesh, True) #write_ascii = True

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


my_lods = lod_mesh_export(bpa_mesh, [100000,50000,10000,1000,100], ".ply", output_path)

# access and visualize 100 triangles
o3d.visualization.draw_geometries([my_lods[1000]])










print("Program terminated successfully...")
































#
#print("Let's draw some primitives")
#mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0,
#                                                height=1.0,
#                                                depth=1.0)
#mesh_box.compute_vertex_normals()
#mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
#mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
#mesh_sphere.compute_vertex_normals()
#mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
#mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.3,
#                                                          height=4.0)
#mesh_cylinder.compute_vertex_normals()
#mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
#mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#    size=0.6, origin=[-2, -2, -2])
#
#print("We draw a few primitives using collection.")
#o3d.visualization.draw_geometries(
#    [mesh_box, mesh_sphere, mesh_cylinder, mesh_frame])
#
#print("We draw a few primitives using + operator of mesh.")
#o3d.visualization.draw_geometries(
#    [mesh_box + mesh_sphere + mesh_cylinder + mesh_frame])
#
#print("Let's draw a cubic using o3d.geometry.LineSet.")
#points = [
#    [0, 0, 0],
#    [1, 0, 0],
#    [0, 1, 0],
#    [1, 1, 0],
#    [0, 0, 1],
#    [1, 0, 1],
#    [0, 1, 1],
#    [1, 1, 1],
#]
#lines = [
#    [0, 1],
#    [0, 2],
#    [1, 3],
#    [2, 3],
#    [4, 5],
#    [4, 6],
#    [5, 7],
#    [6, 7],
#    [0, 4],
#    [1, 5],
#    [2, 6],
#    [3, 7],
#]
#colors = [[1, 0, 0] for i in range(len(lines))]
#line_set = o3d.geometry.LineSet(
#    points=o3d.utility.Vector3dVector(points),
#    lines=o3d.utility.Vector2iVector(lines),
#)
#line_set.colors = o3d.utility.Vector3dVector(colors)
#o3d.visualization.draw_geometries([line_set])
#
#print("Let's draw a textured triangle mesh from obj file.")
#textured_mesh = o3d.io.read_triangle_mesh("../../TestData/crate/crate.obj")
#textured_mesh.compute_vertex_normals()
#o3d.visualization.draw_geometries([textured_mesh])