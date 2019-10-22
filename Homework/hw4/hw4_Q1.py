import numpy as np

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

def rotation_matrix_C(radian_x, radian_y, radian_z):
    Rcx = np.array([[1,0,0],[0,np.cos(radian_x),-np.sin(radian_x)],[0, np.sin(radian_x), np.cos(radian_x)]])
    Rcy = np.array([[np.cos(radian_y),0, np.sin(radian_y)],[0,1,0],[-np.sin(radian_y),0,np.cos(radian_y)]])
    Rcz = np.array([[np.cos(radian_z),-np.sin(radian_z),0],[np.sin(radian_z),np.cos(radian_z),0],[0,0,1]])
    return np.matmul(Rcz, np.matmul(Rcy, Rcx))


if __name__ == "__main__":
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
    three_chosen_points = np.column_stack([house_coords[:,i] for i in [3,5,7]])
    Rcs = [rotation_matrix_C(0, 3*np.pi/4, 0), rotation_matrix_C(-np.pi/2, 0, 0)]
    Centers = [np.array([[-np.sqrt(10), 0, 7+np.sqrt(10)]]), np.array([[0, -10,  0]])]

    M_int = np.array([  [100,    0,  200],
                        [   0, 100,  200],
                        [   0,    0,    1]])

    three_chosen_pts_4by1 = np.append(three_chosen_points, [np.ones((three_chosen_points.shape[1]),  dtype=house_coords.dtype)], 0)
    CA = Camera(Rc=Rcs[0],C=Centers[0], M_int=M_int)
    CB = Camera(Rc=Rcs[1],C=Centers[1], M_int=M_int)
    img_coords_A = np.divide(CA.cam_mat.dot(three_chosen_pts_4by1), CA.cam_mat.dot(three_chosen_pts_4by1)[2])[:-1]
    img_coords_B = np.divide(CB.cam_mat.dot(three_chosen_pts_4by1), CB.cam_mat.dot(three_chosen_pts_4by1)[2])[:-1]
    
    reconstructed_points = []
    for i in range(len(three_chosen_points)):
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

    img_reproj_A = np.divide(CA.cam_mat.dot(reprojected_pts_4by1), CA.cam_mat.dot(reprojected_pts_4by1)[2])[:-1]
    img_reproj_B = np.divide(CB.cam_mat.dot(reprojected_pts_4by1), CB.cam_mat.dot(reprojected_pts_4by1)[2])[:-1]

    error_A = img_reproj_A - img_coords_A
    error_B = img_reproj_B - img_coords_B
    sq_sum_A = sum([error_A[:,i][0]**2+error_A[:,i][1]**2 for i in range(3)])
    sq_sum_B = sum([error_B[:,i][0]**2+error_B[:,i][1]**2 for i in range(3)])

    print("\nSelected camera: (intrinsic matrix, same camera at two different locations)")
    print(M_int)
    print("Corresponding two extrinsic locations: (extrinsic matrix)")
    print(CA.M_ext, CB.M_ext)

    print("\nSelected three points are: (columns)")
    print(three_chosen_pts_4by1[:-1])
    print("Reprojected three points (stereo reconstructed coords): (columns)")
    print(reprojected_pts_4by1[:-1])

    print("\nReprojection Error of Camera A for the selected three points are: (coloumns)")
    print(error_A)
    print("Reprojection Error of Camera B for the selected three points are: (coloumns)")
    print(error_B)

    print("\nReprojection Error sum of square for each selected point at Camera A")
    print(sq_sum_A)
    print("Reprojection Error sum of square for each selected point at Camera B")
    print(sq_sum_B)
    
    