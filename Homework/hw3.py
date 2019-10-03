import numpy as np
# describe a simple house in a selected world coordinate system:
edges = [
    ('a', ['A', 'b', 'e']),
    ('b', ['a', 'B', 'c']),
    ('c', ['e', 'd', 'C', 'b']),
    ('d', ['e', 'D', 'c']),
    ('e', ['a', 'c', 'd', 'E']),
    ('A', ['a', 'B', 'E']),
    ('B', ['A', 'b', 'C']),
    ('C', ['E', 'D', 'c', 'B']),
    ('D', ['E', 'd', 'C']),
    ('E', ['A', 'C', 'D', 'e']),
]
# corresponding coordinates of vertices
coords = [
    ('a', np.array([0, 0, 0])),
    ('b', np.array([8, 0, 0])),
    ('c', np.array([8, 0, 6])),
    ('d', np.array([4, 0, 10])),
    ('e', np.array([0, 0, 6])),
    ('A', np.array([0, 7, 0])),
    ('B', np.array([8, 7, 0])),
    ('C', np.array([8, 7, 6])),
    ('D', np.array([4, 7, 10])),
    ('E', np.array([0, 7, 6])),
]

edge_dict, coords_dict = {}, {}
for (node, connecting_nodes) in edges:
    edge_dict[node] = connecting_nodes
for (node, node_coords) in coords:
    coords_dict[node] = node_coords

house_coords = np.column_stack([n[1] for n in coords])

R_0 = np.array([[],
                [],
                []])

R_1 = np.array([[],
                [],
                []])

R_2 = np.array([[],
                [],
                []])

R_3 = np.array([[],
                [],
                []])

R_4 = np.array([[],
                [],
                []])

t_0 = np.array([[],
                [],
                []])
 
t_1 = np.array([[],
                [],
                []])
 
t_2 = np.array([[],
                [],
                []])
 
t_3 = np.array([[],
                [],
                []])
 
t_4 = np.array([[],
                [],
                []])

M_ext_0 = np.array([[0.707, 0.707, 0, -3], 
                    [-0.707, 0.707, 0, -0.5],
                    [0, 0, 1, 3]])

M_ext_1 = np.array([[0.707, 0.707, 0, -3], 
                    [-0.707, 0.707, 0, -0.5],
                    [0, 0, 1, 3]])

M_ext_2 = np.array([[0.707, 0.707, 0, -3], 
                    [-0.707, 0.707, 0, -0.5],
                    [0, 0, 1, 3]])

M_ext_3 = np.array([[0.707, 0.707, 0, -3], 
                    [-0.707, 0.707, 0, -0.5],
                    [0, 0, 1, 3]])

M_ext_4 = np.array([[0.707, 0.707, 0, -3], 
                    [-0.707, 0.707, 0, -0.5],
                    [0, 0, 1, 3]])

M_int = np.array([  [-100, 0, 200],     #   
                    [-0, -100, 200],    #
                    [0, 0, 1]])         #

# augmented P matrix with extra "1" as the fourth component:
P = np.append(house_coords,
              [np.ones((house_coords.shape[1]), dtype=house_coords.dtype)], 0)
proj_coords = M_int.dot((M_ext.dot(P)))
pixel_coords = np.divide(proj_coords, proj_coords[2])
pixel_coords = np.divide(proj_coords, proj_coords[2])
pixel_coords = pixel_coords[:-1]
projected_dict = {}
for n in range(pixel_coords.shape[1]):
    projected_dict[edges[n][0]] = pixel_coords.T[n]
print(projected_dict)
# using Line2D to plot the projected "house"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
fig = plt.figure()
ax = fig.add_subplot(111)
projected_lines = []

for v in edge_dict.keys():
    for p in edge_dict[v]:
        projected_lines.append(
            Line2D([projected_dict[v][0], projected_dict[p][0]],
                   [projected_dict[v][1], projected_dict[p][1]]))
        ax.add_line(projected_lines[-1])
ax.axis([-100, 400, 0, 500])
plt.show()