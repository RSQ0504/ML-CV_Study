import numpy as np
import cv2
import tqdm
import os
import matplotlib.pyplot as plt
import multiprocessing

def save_points_to_obj(points, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'w') as obj_file:
        for i in range(points.shape[1]):
            x, y, z = points[:, i]
            obj_file.write(f'v {x} {y} {z}\n')


def depth_2_3d(depthmap,P,dot_3d):
    height, width = depthmap.shape
    
    for y in tqdm.tqdm(range(height)):
        for x in range(width):
            d = depthmap[y,x]
            if d == 0:
                continue
            temp_2d = np.array([d*x, d*y, d])
            P_inverse = np.linalg.inv(P[:,:-1]) 
            p_3d = np.dot(P_inverse, (temp_2d - P[:,-1])).reshape((3, 1))
            if dot_3d == []:
                dot_3d = p_3d
            else:
                dot_3d = np.append(dot_3d, p_3d, axis=1)
            # print(dot_3d.shape)
    return dot_3d

def get_camera_parameters(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    result = {}
    for line in lines:
        values = line.strip().split()

        img_name = values[0]

        K = np.array(values[1:10], dtype=float).reshape((3, 3))

        R = np.array(values[10:19], dtype=float).reshape((3, 3))

        t = np.array(values[19:22], dtype=float)

        result[img_name] = {
            'K': K,
            'R': R,
            't': t,
            'Rt': np.column_stack((R, t)),
            'P': np.dot(K,np.column_stack((R, t)))
        }

    return result

def Get3dCoord(q, I0, d):
    x, y = q

    q_new = np.array([d*x, d*y, d])

    P_inverse = np.linalg.inv(I0["P"][:,:-1]) 
    p_3d = np.dot(P_inverse, (q_new - I0["P"][:,-1]))

    return p_3d

def NormalizedCrossCorrelation(C0, C1):
    avg_red_C0 = np.mean(C0[:, 0])
    avg_green_C0 = np.mean(C0[:, 1])
    avg_blue_C0 = np.mean(C0[:, 2])

    C0 -= np.array([avg_red_C0, avg_green_C0, avg_blue_C0])

    l2_norm_C0 = np.linalg.norm(C0)

    C0 /= l2_norm_C0

    avg_red_C1 = np.mean(C1[:, 0])
    avg_green_C1 = np.mean(C1[:, 1])
    avg_blue_C1 = np.mean(C1[:, 2])

    C1 -= np.array([avg_red_C1, avg_green_C1, avg_blue_C1])
    
    l2_norm_C1 = np.linalg.norm(C1)
    
    C1 /= l2_norm_C1

    cross_correlation = np.dot(C0.flatten(), C1.flatten())

    return cross_correlation

def ComputeConsistency(I0, I1, X):
    num_points = X.shape[1]

    X = np.vstack((X, np.ones((1, num_points))))
    #print(I0["P"].shape,X.shape)
    projected_coords_I0 = np.dot(I0["P"], X)
    projected_coords_I1 = np.dot(I1["P"], X)
    #print(I0["mat"].shape)
    
    C0 = np.zeros((num_points, 3))
    C1 = np.zeros((num_points, 3))
    
    for i in range(num_points):
        x0, y0 = projected_coords_I0[:2, i] / projected_coords_I0[2, i]
        x1, y1 = projected_coords_I1[:2, i] / projected_coords_I1[2, i]
        try:
            C0[i] = I0["mat"][int(y0), int(x0)]
        # except Exception as e:
        #     C0[i] = [0,0,0]
        #     # print(X[:,i])
        #     # print(projected_coords_I0[:, i])
        #     # print(y0,x0)
            
        # try:
            C1[i] = I1["mat"][int(y1),int(x1)]
        except Exception as e:
            return -np.inf
            # C1[i] = [0,0,0]
            # print(X[:,i])
            # print(projected_coords_I1[:, i])
            # print(y1,x1)
        
    return NormalizedCrossCorrelation(C0, C1)

def DepthmapAlgorithm(I0, I1, I2, I3, min_depth, max_depth, depth_step, S=5, consistency_threshold=0.7):
    height, width, _ = I0["mat"].shape
    best_depthmap = np.zeros((height, width))
    
    for y in tqdm.tqdm(range(height)):
        for x in range(width):
            if np.all(I0["mat"][y, x] < [30, 30, 30]):
                continue

            best_consistency_score = -np.inf
            best_depth = None

            for d in np.arange(min_depth, max_depth, depth_step):
                # Compute 3D coordinates using Get3dCoord
                X = [] 
                for qx in range(max(0, x - S//2),min(width, x + S//2 + 1)): 
                    for qy in range(max(0, y - S//2), min(height, y + S//2 + 1)):
                        #print(Get3dCoord((qx, qy), I0, d).shape)
                        #print(Get3dCoord((qx, qy), I0, d))
                        # print(qx,qy)
                        # print(Get3dCoord((qx, qy), I0, d))
                        if X == []:
                            X = Get3dCoord((qx, qy), I0, d)
                        else:
                            X = np.column_stack((X,Get3dCoord((qx, qy), I0, d)))

                score01 = ComputeConsistency(I0, I1, X)
                score02 = ComputeConsistency(I0, I2, X)
                score03 = ComputeConsistency(I0, I3, X)
                if score01 == -np.inf or score02 ==-np.inf or score03 == -np.inf:
                    continue
                avg_consistency_score = np.mean([score01, score02, score03])

                # Update best depth if the consistency score is higher
                if avg_consistency_score > best_consistency_score:
                    best_consistency_score = avg_consistency_score
                    best_depth = d

            # Set the depth with the best consistency score for the pixel
            if best_consistency_score >= consistency_threshold:
                best_depthmap[y, x] = best_depth

    return best_depthmap

def generate_depthmap(args):
    idx, img1, img2, img3, img4, min_depth, max_depth, depth_step, S, consistency_threshold = args
    dmap = DepthmapAlgorithm(img1, img2, img3, img4, min_depth, max_depth, depth_step, S=S, consistency_threshold=consistency_threshold)
    np.save(f'../results/my_dmap{idx}.npy', dmap)
    

if __name__ == "__main__":
    result = get_camera_parameters("../data/templeR_par.txt")
    num_ims = len(result)
    images = []
    for index, img in enumerate(result):
        I = {}
        temp_img = cv2.imread(os.path.join("../data",img))
        I["mat"] = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        I["P"] = result[img]["P"]
        images.append(I)
    min_point = np.array([-0.023121, -0.038009, -0.091940])
    max_point = np.array([0.078626, 0.121636, -0.017395])

    corners_3d = np.array([
        [min_point[0], min_point[1], min_point[2]],
        [min_point[0], min_point[1], max_point[2]],
        [min_point[0], max_point[1], min_point[2]],
        [min_point[0], max_point[1], max_point[2]],
        [max_point[0], min_point[1], min_point[2]],
        [max_point[0], min_point[1], max_point[2]],
        [max_point[0], max_point[1], min_point[2]],
        [max_point[0], max_point[1], max_point[2]],
    ])

    projected_points = np.dot(images[0]["P"], np.column_stack((corners_3d, np.ones((8, 1)))).T)
    depth_values = projected_points[-1, :]
    projected = np.zeros((2,projected_points.shape[1]))
    for i in range(projected_points.shape[1]):
        projected[:,i] = projected_points[:2, i] / projected_points[2, i]
        # print(projected[:,i])
        
    color = (255, 0, 0)
    radius = 5
    thickness = -1
    for i in range(projected.shape[1]):
        point_coordinates = (int(projected[0, i]), int(projected[1, i]))
        result_img = cv2.circle(images[0]["mat"], point_coordinates, radius, color, thickness)

    plt.imshow(result_img)
    plt.title('Image with Points')
    plt.savefig('../results/3_5_1(1).png', dpi=300)

    projected_points = np.dot(images[1]["P"], np.column_stack((corners_3d, np.ones((8, 1)))).T)
    depth_values = projected_points[-1, :]
    projected = np.zeros((2,projected_points.shape[1]))
    for i in range(projected_points.shape[1]):
        projected[:,i] = projected_points[:2, i] / projected_points[2, i]
        # print(projected[:,i])
        
    color = (255, 0, 0)
    radius = 5
    thickness = -1
    for i in range(projected.shape[1]):
        point_coordinates = (int(projected[0, i]), int(projected[1, i]))
        result_img = cv2.circle(images[1]["mat"], point_coordinates, radius, color, thickness)

    plt.imshow(result_img)
    plt.title('Image with Points')
    plt.savefig('../results/3_5_1(2).png', dpi=300)

    projected_points = np.dot(images[2]["P"], np.column_stack((corners_3d, np.ones((8, 1)))).T)
    depth_values = projected_points[-1, :]
    projected = np.zeros((2,projected_points.shape[1]))
    for i in range(projected_points.shape[1]):
        projected[:,i] = projected_points[:2, i] / projected_points[2, i]
        # print(projected[:,i])
        
    color = (255, 0, 0)
    radius = 5
    thickness = -1
    for i in range(projected.shape[1]):
        point_coordinates = (int(projected[0, i]), int(projected[1, i]))
        result_img = cv2.circle(images[2]["mat"], point_coordinates, radius, color, thickness)

    plt.imshow(result_img)
    plt.title('Image with Points')
    plt.savefig('../results/3_5_1(3).png', dpi=300)

    projected_points = np.dot(images[3]["P"], np.column_stack((corners_3d, np.ones((8, 1)))).T)
    depth_values = projected_points[-1, :]
    projected = np.zeros((2,projected_points.shape[1]))
    for i in range(projected_points.shape[1]):
        projected[:,i] = projected_points[:2, i] / projected_points[2, i]
        # print(projected[:,i])
        
    color = (255, 0, 0)
    radius = 5
    thickness = -1
    for i in range(projected.shape[1]):
        point_coordinates = (int(projected[0, i]), int(projected[1, i]))
        result_img = cv2.circle(images[3]["mat"], point_coordinates, radius, color, thickness)

    plt.imshow(result_img)
    plt.title('Image with Points')
    plt.savefig('../results/3_5_1(4).png', dpi=300)

    projected_points = np.dot(images[4]["P"], np.column_stack((corners_3d, np.ones((8, 1)))).T)
    depth_values = projected_points[-1, :]
    projected = np.zeros((2,projected_points.shape[1]))
    for i in range(projected_points.shape[1]):
        projected[:,i] = projected_points[:2, i] / projected_points[2, i]
        # print(projected[:,i])
        
    color = (255, 0, 0)
    radius = 5
    thickness = -1
    for i in range(projected.shape[1]):
        point_coordinates = (int(projected[0, i]), int(projected[1, i]))
        result_img = cv2.circle(images[4]["mat"], point_coordinates, radius, color, thickness)

    plt.imshow(result_img)
    plt.title('Image with Points')
    plt.savefig('../results/3_5_1(5).png', dpi=300)
    
    min_depth = np.min(depth_values)
    max_depth = np.max(depth_values)
    depth_step = (max_depth-min_depth)/100
    S = 3
    consistency_threshold = 0.8

    args_list = [
        (1, images[0], images[1], images[3], images[4], min_depth, max_depth, depth_step, S, consistency_threshold),
        # (2, images[1], images[2], images[3], images[4], min_depth, max_depth, depth_step, S, consistency_threshold),
        # (3, images[2], images[0], images[3], images[4], min_depth, max_depth, depth_step, S, consistency_threshold),
        # (4, images[3], images[4], images[0], images[1], min_depth, max_depth, depth_step, S, consistency_threshold),
        # (5, images[4], images[3], images[1], images[2], min_depth, max_depth, depth_step, S, consistency_threshold)
    ]
    
    with multiprocessing.Pool() as pool:
        pool.map(generate_depthmap, args_list)
    
    d1 = np.load("../results/my_dmap1.npy")
    # d2 = np.load("../results/my_dmap2.npy")
    # d3 = np.load("../results/my_dmap3.npy")
    # d4 = np.load("../results/my_dmap4.npy")
    # d5 = np.load("../results/my_dmap5.npy")
    
    gray_image = cv2.cvtColor(images[0]["mat"] , cv2.COLOR_RGB2GRAY)
    d1 = d1 * (gray_image > 50)
    # d2 = d2 * (gray_image > 50)
    # d3 = d3 * (gray_image > 50)
    # d4 = d4 * (gray_image > 50)
    # d5 = d5 * (gray_image > 50)


    plt.figure()
    plt.imshow(d1, cmap='gray')
    plt.axis('image')
    plt.title('Depth Map')
    plt.savefig('../results/3_5(1).png', dpi=300)
    # plt.figure()
    # plt.imshow(d2, cmap='gray')
    # plt.axis('image')
    # plt.title('Depth Map')
    # plt.savefig('../results/3_5(2).png', dpi=300)
    # plt.figure()
    # plt.imshow(d3, cmap='gray')
    # plt.axis('image')
    # plt.title('Depth Map')
    # plt.savefig('../results/3_5(3).png', dpi=300)
    # plt.imshow(d4, cmap='gray')
    # plt.axis('image')
    # plt.title('Depth Map')
    # plt.savefig('../results/3_5(4).png', dpi=300)
    # plt.figure()
    # plt.imshow(d5, cmap='gray')
    # plt.axis('image')
    # plt.title('Depth Map')
    # plt.savefig('../results/3_5(5).png', dpi=300)
    
    dot_3d = []
    dot_3d = depth_2_3d(d1,images[0]["P"],dot_3d)
    # dot_3d = depth_2_3d(d2,images[1]["P"],dot_3d)
    # dot_3d = depth_2_3d(d3,images[2]["P"],dot_3d)
    # dot_3d = depth_2_3d(d4,images[3]["P"],dot_3d)
    # dot_3d = depth_2_3d(d5,images[4]["P"],dot_3d)
    print(dot_3d.shape)
    save_points_to_obj(dot_3d, '../results/output.obj')