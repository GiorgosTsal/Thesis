# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 23:28:02 2021

@author: root
"""



import bpy

# obj = bpy.context.scene.objects.active
# data = obj.data

# # FACES, TRIANGLES
# total_triangles = 0

# for face in data.polygons:
#     vertices = face.vertices
#     triangles = len(vertices) - 2
#     total_triangles += triangles

# print(total_triangles)

# # VERTICES
# vertices = []
# for vertex in data.vertices:
#     pos = [vertex.co[0], vertex.co[1], vertex.co[2]]
#     vertices.append(pos)

# print(vertices)

# # EDGES
# edges = []
# for edge in data.edges:
#     edges.append([edge.vertices[0], edge.vertices[1]])

# print(edges)