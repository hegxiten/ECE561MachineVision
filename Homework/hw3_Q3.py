import numpy as np

# print("Please Input the Degree for θ1 (in DEG):")
# theta_1 = np.float(input())
# print("Please Input the Degree for θ2 (in DEG):")
# theta_2 = np.float(input())
# print("Please Input the Degree for θ3 (in DEG):")
# theta_3 = np.float(input())
# print("Please Input the Degree for θ4 (in DEG):")
# theta_4 = np.float(input())
theta_1, theta_2,theta_3,theta_4 = 30,60,-45,60
radian_1, radian_2, radian_3, radian_4 = \
    theta_1 / 180 * np.pi, theta_2 / 180 * np.pi, theta_3 / 180 * np.pi,\
    theta_4 / 180 * np.pi

L1, L2, L3, L4 = 5,4,3,2


joint_C10 = np.array([[0,0,0]])
joint_C21 = np.array([[L1,0,0]])
joint_C32 = np.array([[L2,0,0]])
joint_C43 = np.array([[L3,0,0]])
joint_C54 = np.array([[L4,0,0]])

def rotation_matrix_C(radian_x, radian_y, radian_z):
    Rcx = np.array([[1, 0, 0], 
                    [0, np.cos(radian_x), -np.sin(radian_x)], 
                    [0, np.sin(radian_x), np.cos(radian_x)]])
    Rcy = np.array([[np.cos(radian_y), 0, np.sin(radian_y)], 
                    [0, 1, 0], 
                    [-np.sin(radian_y), 0, np.cos(radian_y)]])
    Rcz = np.array([[np.cos(radian_z), -np.sin(radian_z), 0],
                    [np.sin(radian_z), np.cos(radian_z), 0], 
                    [0, 0, 1]])
    return np.matmul(Rcz, np.matmul(Rcy, Rcx))

def transformation_affine(Rc, t):    
    M_ext = np.append(Rc.T, (-1)*np.dot(Rc.T, t.T), axis=1)
    return np.append(M_ext, np.array([[0,0,0,1]]), axis=0)

T_10 = transformation_affine(rotation_matrix_C(0,0,radian_1), joint_C10)
T_21 = transformation_affine(rotation_matrix_C(0,0,radian_2), joint_C21)
T_32 = transformation_affine(rotation_matrix_C(0,0,radian_3), joint_C32)
T_43 = transformation_affine(rotation_matrix_C(0,0,radian_4), joint_C43)
T_40 = T_43.dot(T_32.dot(T_21.dot(T_10)))

P_ref = np.linalg.inv(T_40).dot(np.append(joint_C54, np.array([1]).T))
P_ref = np.divide(P_ref, P_ref[-1])

print("1T0 is: ")
print(T_10)
print("2T1 is: ")
print(T_21)
print("3T2 is: ")
print(T_32)
print("4T3 is: ")
print(T_43)
print("4T0 is: ")
print(T_40)
print("Wrist location w.r.t reference frame is: [x, y, z]")
print(P_ref[:-1])

print(np.linalg.inv(T_40).dot(np.array([[2,0,0,1]]).T))