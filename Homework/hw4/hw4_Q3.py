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

pixel_coords = np.array([   
    [200., 200.], 
    [280., 200.], 
    [280., 260.],
    [240., 300.], 
    [200., 260.], 
    [200., 200.],
    [247.05882353, 200.], 
    [247.05882353, 235.29411765],
    [223.52941176, 258.82352941], 
    [200., 235.29411765]
])

# M_int = np.array([[100,    0,  200],
#                   [   0, 100,  200],
#                   [   0,    0,    1]])

# M = [[1.e+02 0.e+00 2.e+02 2.e+03]
#     [0.e+00 1.e+02 2.e+02 2.e+03]
#     [0.e+00 0.e+00 1.e+00 1.e+01]]

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
print(c_transform_w)
