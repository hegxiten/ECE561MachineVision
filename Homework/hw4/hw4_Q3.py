import numpy as np

# print("Enter the number of corresponding-point pairs:")
# N = int(input())

# world_coords = []
# pixel_coords = []

# for i in range(N):
#     world_point = []
#     print("\nEnter the No.{0} point in world coordinates:".format(i))
#     print("\tX (world): ")
#     world_point.append(float(input()))
#     print("\tY (world): ")
#     world_point.append(float(input()))
#     print("\tZ (world): ")
#     world_point.append(float(input()))
#     world_coords.append(world_point)

# for i in range(N):
#     pixel_point = []
#     print("\nEnter the No.{0} point in pixel coordinates:".format(i))
#     print("\tx (pixel): ")
#     pixel_point.append(float(input()))
#     print("\ty (pixel): ")
#     pixel_point.append(float(input()))
#     pixel_coords.append(world_point)

world_coords = np.array([
    [0, 0, 0],
    [8, 0, 0],
    [8, 6, 0],
    [4, 10, 0],
    [0, 6, 0],
    [0, 0, 7],
    [8, 0, 7],
    [8, 6, 7],
    [4, 10, 7],
    [0, 6, 7],
])

pixel_coords = np.array([[
    200.,
    120.,
    120.,
    160.,
    200.,
    200.,
    152.94117647,
    152.94117647,
    176.47058824,
    200.,
],
                         [
                             200., 200., 260., 300., 260., 200., 200.,
                             235.29411765, 258.82352941, 235.29411765
                         ]]).T
print(pixel_coords)
# M_int = np.array([[-100,   0,  200],
#                   [   0, 100,  200],
#                   [   0,   0,    1]])

# M = [ [-1.e+02  0.e+00  2.e+02  2.e+03]
#       [ 0.e+00  1.e+02  2.e+02  2.e+03]
#       [ 0.e+00  0.e+00  1.e+00  1.e+01]]

# refer to http://ksimek.github.io/2012/08/14/decompose/

N = pixel_coords.shape[0]

list_of_2by12_arrays = []
for i in range(N):
    r_1 = list(world_coords[i]) + list([1, 0, 0, 0, 0]) + list([
        (-1) * pixel_coords[i][0] * j for j in world_coords[i]
    ]) + list([(-1) * pixel_coords[i][0]])

    r_2 = list([0, 0, 0, 0]) + list((world_coords[i])) + [1] + list([
        (-1) * pixel_coords[i][1] * j for j in world_coords[i]
    ]) + list([(-1) * pixel_coords[i][1]])
    list_of_2by12_arrays.append(np.array(r_1))
    list_of_2by12_arrays.append(np.array(r_2))

A = np.stack([i for i in list_of_2by12_arrays], axis=0)
_, _, vh = np.linalg.svd(A)
M = vh[:, -1].reshape(3, 4)
print("\nCamera Matrix M")
print(M)

KcRw = M[:, :3]
wRc, K_inv = np.linalg.qr(KcRw)
cRw = np.linalg.inv(wRc)
K = np.linalg.inv(K_inv)
print("\nR")
print(cRw)
print("\nK")
print(K)

_, _, vh_M = np.linalg.svd(M)
wtc = vh_M[:, -1]
wtc = wtc[:3] / wtc[-1]
print("\nCamera origin w.r.t. world")
print(wtc)

ctw = np.asmatrix(-cRw.dot(wtc))
print(cRw)
print("")
print("\nworld origin w.r.t. camera")
print(ctw)
c_transform_w = np.concatenate((cRw, ctw.T), axis=1)

P = np.append(world_coords, np.ones((world_coords.shape[0]), dtype=world_coords.dtype).reshape(1,10).T, axis=1)
P = P.T
print(P)
reprojected_coords = np.matmul(M, P)
pixel_cords = [np.divide(x, x[2])[:-1] for x in [reprojected_coords]]

# draw the five views using matplotlib with keyboard interrupt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
