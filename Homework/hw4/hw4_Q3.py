import numpy as np

print("Enter the number of corresponding-point pairs:")
N = int(input())

world_coords = []
pixel_coords = []

for i in range(N):
    world_point = []
    print("\nEnter the No.{0} point in world coordinates:".format(i))
    print("\tX (world): ")
    world_point.append(float(input()))
    print("\tY (world): ")
    world_point.append(float(input()))
    print("\tZ (world): ")
    world_point.append(float(input()))
    world_coords.append(world_point)

for i in range(N):
    pixel_point = []
    print("\nEnter the No.{0} point in pixel coordinates:".format(i))
    print("\tx (pixel): ")
    pixel_point.append(float(input()))
    print("\ty (pixel): ")
    pixel_point.append(float(input()))
    pixel_coords.append(world_point)

list_of_2by12_arrays = []
for i in range(N):
    r_1 = world_coords[i] + [1,0,0,0,0] + [
        (-1) * pixel_coords[i][0] * j for j in world_coords[i]
        ] + [(-1) * pixel_coords[i][0]]
    
    r_2 = [0,0,0,0] + world_coords[i] + [1] + [
        (-1) * pixel_coords[i][1] * j for j in world_coords[i]
        ] + [(-1) * pixel_coords[i][1]]
    list_of_2by12_arrays.append(np.array(r_1))
    list_of_2by12_arrays.append(np.array(r_2))

A = np.stack([i for i in list_of_2by12_arrays],axis=0)

_, _, vh = np.linalg.svd(A)
M = vh[:,-1].reshape(3,4)
print(M)

_, _, vh_M = np.linalg.svd(M)
wTc = vh_M[:,-1]
print(wTc)

b = M[:, -1].copy()
x = np.linalg.lstsq(M[:, :-1], -b)[0]
x = np.r_[x,1]
print(x)


R_inv, K_inv = np.linalg.qr(M)
print(R_inv)
print(K_inv)

# b = A[:, -1].copy()

# A_1 = A[:, :-1]
# A_t=np.transpose(A_1)
# A_tA_1 = np.linalg.inv(A_t.dot(A_1))
# q = (A_tA_1.dot(A_t)).dot(b)
# print(q)

# pivoting?
#

# x = np.linalg.svd(A[:, :-1], -b, rcond=None)[0]
# print(x)
# M = x.reshape(3,4)
# print(M)