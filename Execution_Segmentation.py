import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom
from data_processing_utils import read_dicom,\
    process_xray, read_pts, extract_patches,f_2_b, pad,worker
import argparse
import glob
from scipy.ndimage import interpolation
from PIL import Image
import matplotlib
from read_json import Read_json
from standard_ROI import standard_ROI_amir, show_patches,fixed_roi
from polygan import polylines_amir
from data_processing_utils import rotate, rotate_amir
import os.path
def Execution_Segmentation_amir(list_ID):

    dir_t = '/via_export_json (1).json'
    dir_f = '/via_export_json.json'




    if list_ID['Label'] == 1:
        prefix_ann = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/OA/'
        prefix = 'C:/Users/Amir Kazemtarghi/Documents/data/5slice/OA/'
    else:
        prefix_ann = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/No-OA/'
        prefix = 'C:/Users/Amir Kazemtarghi/Documents/data/5slice/no-OA/'




    #image_dir_f = '/' + list_ID['SeriesDescription'] + '_femur_.png'
    image_dir_t = '/' + list_ID['SeriesDescription']

    im_fold = prefix_ann + str(list_ID['ParticipantID']) + '/via_export_json (2).json'

    if os.path.exists(im_fold):
        if list_ID['SeriesDescription'] == 'SAG_3D_DESS_RIGHT':
            dir_t = '/via_export_json (3).json'
            dir_f = '/via_export_json (2).json'


    list_x_tibia, list_y_tibia = Read_json(prefix_ann + str(list_ID['ParticipantID']) + dir_t)
    list_x_femur, list_y_femur = Read_json(prefix_ann + str(list_ID['ParticipantID']) + dir_f)

    check_a = prefix + str(list_ID['ParticipantID']) + image_dir_t + '_tibia_' + 'C' + '.png'

    C = cv2.imread(prefix + str(list_ID['ParticipantID']) + image_dir_t + '_tibia_' + '_C' + '.png', 0)
    # N = cv2.imread(prefix + str(list_ID['ParticipantID']) + image_dir_t + '_tibia_' + '_N' + '.png', 0)
    # P = cv2.imread(prefix + str(list_ID['ParticipantID']) + image_dir_t + '_tibia_' + 'p' + '.png', 0)
    # NN = cv2.imread(prefix + str(list_ID['ParticipantID']) + image_dir_t + '_tibia_' + 'NN' + '.png', 0)
    # PP = cv2.imread(prefix + str(list_ID['ParticipantID']) + image_dir_t + '_tibia_' + 'PP' + '.png', 0)

    # determining the center of joint
    x = (list_x_tibia[8] + list_x_tibia[9])//2
    y = (list_y_tibia[8] + list_y_tibia[9]) // 2

    center = np.array([x, y])
    p1 = np.array([list_x_tibia[5], list_y_tibia[5]])
    p2 = np.array([list_x_tibia[12], list_y_tibia[12]])

    points_t = np.zeros([18, 2])
    points_t[:, 0] = list_x_tibia
    points_t[:, 1] = list_y_tibia

    # if list_ID['ParticipantID'] == 9267648:
    #     implot = plt.imshow(C)




    points_f = np.zeros([17, 2])
    points_f[:, 0] = list_x_femur
    points_f[:, 1] = list_y_femur

    rotated_C, M = rotate(C, center, p1, p2)
    # rotated_N, M = rotate(N, center, p1, p2)
    # rotated_P, M = rotate(P, center, p1, p2)
    # rotated_NN, M = rotate(NN, center, p1, p2)
    # rotated_PP, M = rotate(PP, center, p1, p2)
    #rotated_image_femur, M = rotate(mri_image_femur, center, p1, p2)

    rotated_landmarks_t = rotate_amir(points_t, M)
    rotated_landmarks_f = rotate_amir(points_f, M)

    image_mask_tibia = polylines_amir(rotated_landmarks_t, rotated_C)
    # image_mask_femur = polylines_amir(rotated_landmarks_f, mri_image_femur)



    roi_med_t, roi_lat_t, r, m, l = fixed_roi(image_mask_tibia,
                                              rotated_landmarks_t[12, :],
                                              rotated_landmarks_t[5, :],
                                              rotated_landmarks_t[9, :],
                                              rotated_landmarks_t[11, :],
                                              rotated_landmarks_t[6, :]

                                              )

    # roi_med_f, roi_lat_f, r, box_med_f, box_lat_f = standard_ROI_amir(image_mask_femur,
    #                                                                   rotated_landmarks_f[:, 0],
    #                                                                   rotated_landmarks_f[:, 1])
    #
    # if list_ID['ParticipantID'] == 9081858:
    #     implot = plt.imshow(C)
    #
    #     # put a blue dot at (10, 20)
    #     plt.scatter([10], [20])
    #
    #     # put a red dot, size 40, at 2 locations:
    #     plt.scatter(x=points_t[:, 0], y=points_t[: , 1], c='r', s=40)
    #
    #     plt.show()


    # show_patches(rotated_C, r , m, l, list_ID['ParticipantID'])
    #     show_patches(rotated_N, r, m, l, list_ID['ParticipantID'])
    #     show_patches(rotated_P, r, m, l, list_ID['ParticipantID'])
    #
    #     a = 1



    if (len(roi_lat_t[roi_lat_t==0]) < 10000) & (len(roi_med_t[roi_med_t == 0]) < 1000000):

        roi_med_C, roi_lat_C, r, m, l = fixed_roi(rotated_C,
                                                  rotated_landmarks_t[12, :],
                                                  rotated_landmarks_t[5, :],
                                                  rotated_landmarks_t[9, :],
                                                  rotated_landmarks_t[11, :],
                                                  rotated_landmarks_t[6, :])



        # roi_med_t = np.zeros((roi_med_C.shape[0], roi_med_C.shape[1], 3))
        # roi_lat_t = np.zeros((roi_lat_C.shape[0], roi_lat_C.shape[1], 3))
        #
        # roi_med_t[:, :, 0] = roi_med_C
        # roi_med_t[:, :, 1] = roi_med_N
        # roi_med_t[:, :, 2] = roi_med_P
        #
        # roi_lat_t[:, :, 0] = roi_lat_C
        # roi_lat_t[:, :, 1] = roi_lat_N
        # roi_lat_t[:, :, 2] = roi_lat_P

        return roi_med_C, roi_lat_C

    else:
        #show_patches(mri_image_tibia, box_lat_t, box_med_t, i, r)
        print('Not correct lateral')
        print(list_ID['ParticipantID'])





















