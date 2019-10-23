import numpy as np
def rotation_matrix_C(radian_x, radian_y, radian_z):
    Rcx = np.array([[1,0,0],[0,np.cos(radian_x),-np.sin(radian_x)],[0, np.sin(radian_x), np.cos(radian_x)]])
    Rcy = np.array([[np.cos(radian_y),0, np.sin(radian_y)],[0,1,0],[-np.sin(radian_y),0,np.cos(radian_y)]])
    Rcz = np.array([[np.cos(radian_z),-np.sin(radian_z),0],[np.sin(radian_z),np.cos(radian_z),0],[0,0,1]])
    return np.matmul(Rcz, np.matmul(Rcy, Rcx))

# describe a simple house in a selected world coordinate system:
edges = [
    ('a', ['A', 'b', 'e']),
    ('b', ['a', 'B', 'c']),
    ('c', ['d', 'C', 'b']),
    ('d', ['e', 'D', 'c']),
    ('e', ['a', 'd', 'E']),
    ('A', ['a', 'B', 'E']),
    ('B', ['A', 'b', 'C']),
    ('C', ['D', 'c', 'B']),
    ('D', ['E', 'd', 'C']),
    ('E', ['A', 'D', 'e']),
]
# corresponding coordinates of vertices in the simple house
coords = [
    ('a', np.array([0,  0,  0])),
    ('b', np.array([8,  0,  0])),
    ('c', np.array([8,  6,  0])),
    ('d', np.array([4,  10, 0])),
    ('e', np.array([0,  6,  0])),
    ('A', np.array([0,  0,  7])),
    ('B', np.array([8,  0,  7])),
    ('C', np.array([8,  6,  7])),
    ('D', np.array([4,  10, 7])),
    ('E', np.array([0,  6,  7])),
]

edge_dict, coords_dict = {}, {}
for (node, connecting_nodes) in edges:
    edge_dict[node] = connecting_nodes
for (node, node_coords) in coords:
    coords_dict[node] = node_coords

house_coords = np.column_stack([n[1] for n in coords])

M_int = np.array([[-100,   0,  200],
                  [   0, 100,  200],
                  [   0,    0,   1]])
# refer to http://ksimek.github.io/2012/08/14/decompose/

Rc_0 = rotation_matrix_C(0, 0, 0)
C_0 = np.array([[0, 0, -10]])
M_ext_0 = np.append(Rc_0.T, -np.dot(Rc_0.T, C_0.T), axis=1)

Rc_1 = rotation_matrix_C(0, 3*np.pi/4, 0)
C_1 = np.array([[-np.sqrt(10), 0, 7+np.sqrt(10)]])
M_ext_1 = np.append(Rc_1.T, -np.dot(Rc_1.T, C_1.T), axis=1)

Rc_2 = rotation_matrix_C(0, 5*np.pi/4, 0)
C_2 = np.array([[8+np.sqrt(10), np.sqrt(10), 7+np.sqrt(10)]])
M_ext_2 = np.append(Rc_2.T, -np.dot(Rc_2.T, C_2.T), axis=1)

Rc_3 = rotation_matrix_C(-np.pi/2, 0, 0)
C_3 = np.array([[0, -10,  0]])
M_ext_3 = np.append(Rc_3.T, -np.dot(Rc_3.T, C_3.T), axis=1)

Rc_4 = rotation_matrix_C(np.pi/2, 0, 0)
C_4 = np.array([[0, 15,  0]])
M_ext_4 = np.append(Rc_4.T, -np.dot(Rc_4.T, C_4.T), axis=1)

# augmented P matrix with extra "1" as the fourth component:
P = np.append(house_coords, [np.ones((house_coords.shape[1]),  dtype=house_coords.dtype)], 0)

proj_cords = [np.matmul(M_int, np.matmul(x, P)) for x in [M_ext_0, M_ext_1, M_ext_2, M_ext_3, M_ext_4]]
pixel_cords = [np.divide(x, x[2])[:-1] for x in proj_cords]
print(np.matmul(M_int, M_ext_0))
print(pixel_cords[0])
# draw the five views using matplotlib with keyboard interrupt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

for view in pixel_cords:
    projected_dict = {}
    for n in range(view.shape[1]):
        projected_dict[edges[n][0]] = view.T[n]
    # using Line2D to plot the projected "house"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    projected_lines = []
    for v in edge_dict.keys():
        for p in edge_dict[v]:
            projected_lines.append(
                Line2D([projected_dict[v][0], projected_dict[p][0]],
                       [projected_dict[v][1], projected_dict[p][1]]))
            ax.add_line(projected_lines[-1])
    ax.autoscale(enable = True)
    plt.draw()

    plt.waitforbuttonpress(0)
    plt.close(fig)