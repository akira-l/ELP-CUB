import numpy
import numpy as np
import torch

import pdb


def procrustes(X, Y, scaling=True, reflection='best'):
    n, m = X.shape
    ny, my = Y.shape
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()
    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY
    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)
    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    if reflection is not 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0
        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)
    traceTA = s.sum()
    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY
        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2
        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX
    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)
    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}
    return d, Z, tform


def transfer_coord(pred_point, bank_point, pick_num):
    pred_point = (np.asarray(pred_point)).astype(np.float64)
    bank_point = (np.asarray(bank_point)).astype(np.float64)
    d, Z_pts, Tform = procrustes(pred_point, bank_point)
    
    R = np.eye(3)
    R[0:2, 0:2] = Tform['rotation']
    S = np.eye(3)*Tform['scale']
    S[2, 2] = 1
    t = np.eye(3)
    t[0:2, 2] = Tform['translation']
    M = np.dot(np.dot(R, S), t.T).T
    pad_point = np.hstack(
         (pred_point, np.array(([[1 for i in range(pick_num)]])).T))
    re_point = np.dot(M, pad_point.T).T 
    return re_point
    #return re_point[:, :2].astype(int)


def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])







if __name__ == '__main__':

    ''' 
    import cv2
    import dlib

    img = cv2.imread("./img_10.jpg")
    img = cv2.resize(img, (500,500))
    image = cv2.imread("./img_7.jpg")
    image = cv2.resize(image, (400,400))
    #ref_point_file = 'ref_points.txt'
    #ref_point = get_ref_point(ref_point_file)
    # print(ref_point)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    
    face = detector(image, 1)
    landmarks = ([(p.x, p.y) for p in predictor(image, face[0]).parts()])
    #print(landmarks)
    #cv.waitKey(0)
    avg_landmarks = []
    for point in landmarks:
        x = point[0]
        y = point[1]
        avg_landmarks.append((x, y))
    avg_landmarks = np.asarray(avg_landmarks)
    #ref_landmarks = np.asarray(ref_point)
    pdb.set_trace()
    M = get_M(avg_landmarks, avg_landmarks, 2)
    avg_landmarks = np.hstack(
        (avg_landmarks, np.array(([[1 for i in range(0, len(landmarks))]])).T))
    clandmarks = np.dot(M, avg_landmarks.T).T
    flandmarks = list((point[0], point[1]) for point in clandmarks)
    print(flandmarks)
    
    for point in flandmarks:
        pos = (int(point[0]), int(point[1]))
        print(pos)
        cv.circle(image, pos, 1, (0,0,0), -1)
    # cv.namedWindow("tezhengdian", 0)
    cv.imshow("yuantu", img)
    cv.imshow("image", image)
    #cv.waitKey(0)
    cv.imwrite(path, image)
    '''
    

    pick_num = 10
    feat = torch.rand(4, 16*16)
    bank = torch.rand(4, 16*16)
    _, feat_pos = torch.topk(feat[0], pick_num)
    feat_pos_x = feat_pos // 16 
    feat_pos_y = feat_pos % 16 
    feat_pos_coord = torch.stack((feat_pos_x, feat_pos_y))

    _, bank_pos = torch.topk(bank[0], pick_num)
    bank_pos_x = bank_pos // 16 
    bank_pos_y = bank_pos % 16
    bank_pos_coord = torch.stack((bank_pos_x, bank_pos_y))

    matrix = transfer_coord(feat_pos_coord.permute(1, 0), bank_pos_coord.permute(1, 0))
    pdb.set_trace()






