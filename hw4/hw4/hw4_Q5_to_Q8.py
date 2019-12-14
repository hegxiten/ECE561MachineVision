# using the same polygon house that used in both HW3_Q1 and HW4_Q4

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

# left camera
Rc_0 = rotation_matrix_C(0, 0, 0)
C_0 = np.array([[0, 0, -10]])
M_ext_0 = np.append(Rc_0.T, -np.dot(Rc_0.T, C_0.T), axis=1)

# right camera 
Rc_1 = rotation_matrix_C(0, 3*np.pi/4, 0)
C_1 = np.array([[-np.sqrt(10), 0, 7+np.sqrt(10)]])
M_ext_1 = np.append(Rc_1.T, -np.dot(Rc_1.T, C_1.T), axis=1)

# augmented P matrix with extra "1" as the fourth component:
P = np.append(house_coords, [np.ones((house_coords.shape[1]),  dtype=house_coords.dtype)], axis=0)

proj_cords = [np.matmul(M_int, np.matmul(x, P)) for x in [M_ext_0, M_ext_1]]
pixel_cords = [np.divide(x, x[2])[:-1] for x in proj_cords]

# two image coordinates, shape = (2,10) for each
img_L, img_R = pixel_cords[0].T, pixel_cords[1].T

list_of_Nby9_input_arrays = []
for i in range(img_L.shape[0]):
    r = np.append(
        img_L[i, 0] * img_R[i, 0], (
        img_L[i, 0] * img_R[i, 1], 
        img_L[i, 0], 
        img_L[i, 1] * img_R[i, 0], 
        img_L[i, 1] * img_R[i, 1], 
        img_L[i, 1], 
        img_R[i, 0],
        img_R[i, 1],
        1.0)
        )
    list_of_Nby9_input_arrays.append(r)
A = np.asmatrix(list_of_Nby9_input_arrays)

_, _, vh = np.linalg.svd(A)
last_col = np.squeeze(np.asarray(vh[:, -1]))
print(last_col.shape)
normalized_last_col = last_col / np.sqrt(np.sum(last_col**2))
F = normalized_last_col.reshape(3, 3).T
print("\nNormalized Fundamental Matrix  F")
print(F)
E = np.linalg.inv(M_int).T.dot(F).dot(M_int)
print("\nEssential Matrix")
print(E)

W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

U, S, Ve = np.linalg.svd(E)
S1 = - U.dot(Z).dot(U.T)
S2 = U.dot(Z).dot(U.T)
R1 = U.dot(W.T).dot(Ve.T)
R2 = U.dot(W).dot(Ve.T)

foundit = 0

p_l = proj_cords[0] # 3 by 10

# case 1
if (not foundit):
    S = S1
    R = R1
    tlr = np.array([S[2,1], S[0,2], S[1,0]]).T
    p_r = R.dot(np.subtract(p_l, tlr.reshape(3,1)))
    print(p_r[2])
    if (min(p_r[2] )>0): 
        foundit = 1

# case 2
if (not foundit):
    S = S2
    R = R1
    tlr = np.array([S[2,1], S[0,2], S[1,0]]).T
    p_r = R.dot(np.subtract(p_l, tlr.reshape(3,1)))
    print(p_r[2])
    if (min(p_r[2] )>0): 
        foundit = 1

# case 3
if (not foundit):
    S = S1
    R = R2
    tlr = np.array([S[2,1], S[0,2], S[1,0]]).T
    p_r = R.dot(np.subtract(p_l, tlr.reshape(3,1)))
    print(p_r[2])
    if (min(p_r[2] )>0): 
        foundit = 1

# case 4
if (not foundit):
    S = S2
    R = R2
    tlr = np.array([S[2,1], S[0,2], S[1,0]]).T
    p_r = R.dot(np.subtract(p_l, tlr.reshape(3,1)))
    print(p_r[2])
    if (min(p_r[2] )>0): 
        foundit = 1


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d

left_pixels = np.divide(p_l, p_l[2])[:-1]
right_pixels = np.divide(p_r, p_r[2])[:-1]
views_pixel_cords = [left_pixels, right_pixels]

# Question 7: 3D reconstruction using E

for view in views_pixel_cords:
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

for view in [p_l, p_r]:
    projected_dict = {}
    for n in range(view.shape[1]):
        projected_dict[edges[n][0]] = view.T[n]
    # using Line3D to plot the reconstructed "house"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    projected_lines = []
    for v in edge_dict.keys():
        for p in edge_dict[v]:
            ax.plot3D(  [projected_dict[v][0], projected_dict[p][0]],
                        [projected_dict[v][1], projected_dict[p][1]],
                        [projected_dict[v][2], projected_dict[p][2]],color = 'b')
            
    ax.autoscale(enable = True)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

# Question8: 3D reconstruction using Euclidian 
# Borrowed codes from hw4_Q1.py
class Camera(object):

    def __init__(self, *args, **kwargs):
        self.Rc = kwargs.get("Rc")
        self.C = kwargs.get("C")
        self.M_int = kwargs.get("M_int")
        
    @property
    def M_ext(self): return np.append(self.Rc.T, -np.dot(self.Rc.T, self.C.T), axis=1)
            
    @property
    def cam_mat(self): return np.matmul(self.M_int, self.M_ext)
    
    @property
    def m_0_T(self): return self.cam_mat[0]

    @property
    def m_1_T(self): return self.cam_mat[1]

    @property
    def m_2_T(self): return self.cam_mat[2]

CA = Camera(Rc=Rc_0, C=C_0, M_int=M_int)
CB = Camera(Rc=Rc_1, C=C_1, M_int=M_int)
chosen_pts_4by1 = np.append(house_coords, [np.ones((house_coords.shape[1]),  dtype=house_coords.dtype)], 0)
img_coords_A = np.divide(CA.cam_mat.dot(chosen_pts_4by1), CA.cam_mat.dot(chosen_pts_4by1)[2])[:-1]
img_coords_B = np.divide(CB.cam_mat.dot(chosen_pts_4by1), CB.cam_mat.dot(chosen_pts_4by1)[2])[:-1]

reconstructed_points = []
for i in range(chosen_pts_4by1.shape[1]):
    A_col_0 = img_coords_A[:,i][0]*CA.m_2_T-CA.m_0_T
    A_col_1 = img_coords_A[:,i][1]*CA.m_2_T-CA.m_1_T
    A_col_2 = img_coords_B[:,i][0]*CB.m_2_T-CB.m_0_T
    A_col_3 = img_coords_B[:,i][1]*CB.m_2_T-CB.m_1_T
    A = np.stack((A_col_0,A_col_1,A_col_2,A_col_3),axis=0)
    b = A[:, -1].copy()
    x = np.linalg.lstsq(A[:, :-1], -b)[0]
    x = np.r_[x,1]
    reconstructed_points.append(x)

reprojected_pts_4by1 = np.array(reconstructed_points).T
reconstructed_points = np.array(reconstructed_points)

# left and right images
for view in [img_coords_A, img_coords_B]:
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

recon_3d = reconstructed_points.T[:-1,:]
# Euclidian 3D plotting
for view in [recon_3d]:
    projected_dict = {}
    print(view.shape)
    for n in range(view.shape[1]):
        projected_dict[edges[n][0]] = view.T[n]
    # using Line3D to plot the reconstructed "house"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    projected_lines = []
    for v in edge_dict.keys():
        for p in edge_dict[v]:
            ax.plot3D(  [projected_dict[v][0], projected_dict[p][0]],
                        [projected_dict[v][1], projected_dict[p][1]],
                        [projected_dict[v][2], projected_dict[p][2]],color = 'b')
            
    ax.autoscale(enable = True)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)