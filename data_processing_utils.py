"""
A module to pre-process and read knee x-rays

(c) Aleksei Tiulpin, University of Oulu, 2016-2018.

"""

import pydicom as dicom
import numpy as np
import cv2
import os

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import scipy.ndimage
def resample(image, scan, new_spacing=[0.2, 0.2]):
    # Determine current pixel spacing
    spacing = map(float, (scan.PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing

def read_dicom(filename):
    """
    Reads a dicom file

    Parameters
    ----------
    filename : str
        Full path to the image

    Returns
    -------
    out : tuple
        Image itself as uint16, spacing, and the DICOM metadata

    """

    try:
        data = dicom.read_file(filename)
    except:
        return None
    img = np.frombuffer(data.PixelData, dtype=np.uint16).copy().astype(np.float64)

    if data.PhotometricInterpretation == 'MONOCHROME1':
        img = img.max() - img
    try:
        img = img.reshape((data.Rows, data.Columns))
    except:
        return None

    try:
        if isinstance(data.ImagerPixelSpacing, str):
            data.ImagerPixelSpacing = data.ImagerPixelSpacing.split()
    except:
        pass

    try:
        if isinstance(data.PixelSpacing, str):
            data.PixelSpacing = data.PixelSpacing.split()
    except:
        pass

    try:
        return img, float(data.ImagerPixelSpacing[0]), data
    except:
        pass
    try:
        return img, float(data.PixelSpacing[0]), data
    except:
        return None


def process_xray(img, cut_min=5, cut_max=99, multiplier=255):
    """
    Processes an X-ray image by truncating the histogram and
    applying a global contrast normalization (GCN) to it.

    Parameters
    ----------
    img : ndarray
        Input image
    cut_min : int or float
        Lowest percentile to truncate the histogram.
    cut_max : in or float
        Highest percentile to truncate the histogram
    multiplier : a multiplication factor to use after GCN

    Returns
    -------
    out : ndarray
        A preprocessed image in np.float64.

    """
    # This function changes the histogram of the image by doing global contrast normalization
    # cut_min - lowest percentile which is used to cut the image histogram
    # cut_max - highest percentile

    img = img.copy().astype(np.float64)
    lim1, lim2 = np.percentile(img, [cut_min, cut_max])
    img[img < lim1] = lim1
    img[img > lim2] = lim2

    img -= lim1
    img /= img.max()
    img *= multiplier

    return img


def read_pts(fname):
    """
    Reads the points annotated using BineFinder.

    Parameters
    ----------
    fname : full path to the annotation

    Returns
    -------
    out : ndarray
        Nx2 ndarray of (x,y) landmark points

    """
    with open(fname) as f:
        content = f.read()
    arr = np.array(list(map(lambda x: [float(x.split()[0]), float(x.split()[1])], content.split('\n')[3:-2])))
    return arr


def f_2_b(img):
    """
    Converts a float [0..255] image into [0, 255] uint8.
    First teh function rounds the image and then clips it.

    Parameters
    ----------
    img : ndarray
        Input image

    Returns
    -------
    out : ndarray
        Results

    """
    img = np.round(img)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def pad(img, padding):
    """
    Pads an image with zero padding from each side.

    Parameters
    ----------
    img : ndarray
        Input image
    padding : int
        Padding

    Returns
    -------
    out : ndarray
        Padded image

    """
    row, col = img.shape
    tmp = np.zeros((row + 2 * padding, col + 2 * padding), dtype=img.dtype)
    tmp[padding:padding + row, padding:padding + col] = img
    return tmp


def rotate(img, c, p1, p2):
    """
    Rotates an image around the point c without changing the size of the coordinate frame
    so that the line going through the points p1, p2 is horizontal.
    Used for teh alignment of tibial plateau.

    Parameters
    ----------
    img : ndarray
    c : tuple or or list or ndarray
        Center of rotation
    p1 : tuple or or list or ndarray
        First point on the line
    p2 : tuple or or list or ndarray
        Second point on teh line
    Returns
    -------
    out : tuple
        Image and the corresponding rotation matrix
    """
    # Based on https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    W = img.shape[0]
    ang = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    M = cv2.getRotationMatrix2D((c[0], c[1]), ang, 1)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((W * sin) + (W * cos))
    nH = int((W * cos) + (W * sin))

    img = cv2.warpAffine(img, M, (nW, nH), cv2.INTER_CUBIC)
    return img, M


def get_ost_jsw_patches(I, p1, p2, f_p1, f_p2, size_w, size_h, return_rects=False):
    """
    Cuts the medial and the lateral patches of size_h x size_w mm from the
    the knee image. The center of the patches will be the mean of tibial and the lateral landmarks.

    Parameters
    ----------
    I : ndarray
        Input image
    p1 : tuple or ndarray
        Tibal landmark (lateral)
    p2 : tuple or ndarray
        Tibial landmark (medial)
    f_p1 : tuple or ndarray
        Femoral landmark (lateral)
    f_p2 : tuple or ndarray
        Femoral landmark (medial)
    size_w : int
        Width in mm
    size_h : int
        Height in mm.
    return_rects : bool
        Whether to return bounding rects in the format (x0, y0, x1, y1) together with the images

    Returns
    -------
    out : tuple
        Tuple, which contains two images and their bounding boxes in the following format:
        image lateral, image medial, lateral rect, medial rect.
        If return_rects is False, then only the images are returned.

    """
    p1 = np.array([p1[0], (p1[1] + f_p1[1]) // 2])
    p2 = np.array([p2[0], (p2[1] + f_p2[1]) // 2])

    rect_l = [p1[0] - size_w // 2, p1[1] - size_h // 2, p1[0] + size_w // 2, p1[1] + size_h // 2]

    rect_m = [p2[0] - size_w // 2, p2[1] - size_h // 2, p2[0] + size_w // 2, p2[1] + size_h // 2]

    L = I[rect_l[1]:rect_l[3], rect_l[0]:rect_l[2]]
    M = I[rect_m[1]:rect_m[3], rect_m[0]:rect_m[2]]
    if return_rects:
        return L, M, rect_l, rect_m
    else:
        return L, M
def rotate_amir(coordinates,M):

    import pandas as pd
    list= []
    for i in range(len(coordinates)):
        dot = coordinates[i, :]
        dot = np.dot(M, np.array(dot.tolist() + [1])).astype(int)
        list.append(dot)
    output = np.vstack(list)

    return output




def extract_patches(I, landmarks,landmarks2, spacing, sizemm, o_sizemm_w, o_sizemm_h):
    """
    Extracts full images of left and right knees, as well as the side patches.
    Such patches are cropped for the exactly the same anatomical locations.

    Parameters
    ----------
    I : ndarray
        Input bilateral image
    spacing : float
        Image spacing
    sizemm : int
        Size of the ROI to crop (in mm)
    landmarks : dict
        Landmarks produced using Bone Finder and stored into dict per bone per knee.
    o_sizemm_w : int
        Width of the side osteophyte patch
    o_sizemm_h : int
        Width of the side osteophyte patch

    Returns
    -------
    out : tuple
        The result containes the tuple of dictionaries.
        One corresponds to the full image, then side patches and then
        the meta-information, which can be further used: bounding boxes and the rotation matrices.

    """
    # Width of the ROI
    W = int(sizemm / spacing)

    # size of the osteophyte patches in pixels
    o_size_w = int(o_sizemm_w / spacing)
    o_size_h = int(o_sizemm_h / spacing)

    # Getting the corner points from the landmarks
    p1r = landmarks['TR'][0, :].copy()
    p1l = landmarks['TL'][0, :].copy()

    p2r = landmarks['TR'][-1, :].copy()
    p2l = landmarks['TL'][-1, :].copy()

    f_p1r = landmarks['FR'][0, :].copy()
    f_p1l = landmarks['FL'][0, :].copy()

    f_p2r = landmarks['FR'][-1, :].copy()
    f_p2l = landmarks['FL'][-1, :].copy()

    # Determining the center of the joint using tibial landmarks
    Lcx, Lcy = landmarks['TL'][landmarks['TL'].shape[0] // 2, :].astype(int)
    Rcx, Rcy = landmarks['TR'][landmarks['TR'].shape[0] // 2, :].astype(int)

    # Defing the bounding boxes
    bboxR = [Rcx - W // 2, Rcy - W // 2, Rcx + W // 2, Rcy + W // 2]
    bboxL = [Lcx - W // 2, Lcy - W // 2, Lcx + W // 2, Lcy + W // 2]

    # Cropping the ROIs (left flipped)
    RR = I[bboxR[1]:bboxR[3], bboxR[0]:bboxR[2]]
    LL = cv2.flip(I[bboxL[1]:bboxL[3], bboxL[0]:bboxL[2]], 1)

    # Adjusting the points to the new coordinate frame,
    # which is within each bounding box.
    cr = np.array([Rcx, Rcy])
    cl = np.array([Lcx, Lcy])

    p1r -= bboxR[:2]
    p2r -= bboxR[:2]
    p1l -= bboxL[:2]
    p2l -= bboxL[:2]

    f_p1r -= bboxR[:2]
    f_p2r -= bboxR[:2]
    f_p1l -= bboxL[:2]
    f_p2l -= bboxL[:2]

    cr -= bboxR[:2]
    cl -= bboxL[:2]

    p1l[0] = W - p1l[0]
    p2l[0] = W - p2l[0]
    f_p1l[0] = W - f_p1l[0]
    f_p2l[0] = W - f_p2l[0]

    # Rotating the joints for normalization
    L, M1 = rotate(LL, cl, p1l, p2l)
    R, M2 = rotate(RR, cr, p1r, p2r)

    landmarks2['L'] -= bboxL[:2]
    landmarks2['L'][:, 0] = W -landmarks2['L'][:, 0]
    rotated_landmarks2 = rotate_amir(landmarks2['L'], M1)

    landmarks2['R'] -= bboxR[:2]
    rotated_landmarks1 = rotate_amir(landmarks2['R'], M2)



    # Rotating the corresponding points
    p1l = np.dot(M1, np.array(p1l.tolist() + [1])).astype(int)
    p2l = np.dot(M1, np.array(p2l.tolist() + [1])).astype(int)

    p1r = np.dot(M2, np.array(p1r.tolist() + [1])).astype(int)
    p2r = np.dot(M2, np.array(p2r.tolist() + [1])).astype(int)

    f_p1l = np.dot(M1, np.array(f_p1l.tolist() + [1])).astype(int)
    f_p2l = np.dot(M1, np.array(f_p2l.tolist() + [1])).astype(int)

    f_p1r = np.dot(M2, np.array(f_p1r.tolist() + [1])).astype(int)
    f_p2r = np.dot(M2, np.array(f_p2r.tolist() + [1])).astype(int)








    right_lateral, right_medial, = get_ost_jsw_patches(R, p1r, p2r, f_p1r, f_p2r, o_size_w, o_size_h)
    left_lateral, left_medial, = get_ost_jsw_patches(L, p1l, p2l, f_p1l, f_p2l, o_size_w, o_size_h)

    full_images = {'R': R, 'L': L}
    side_patches = {'R': {'lat': f_2_b(right_lateral), 'med': f_2_b(right_medial)},
                    'L': {'lat': f_2_b(left_lateral), 'med': f_2_b(left_medial)}}

    metadata = {'R': {'bbox': bboxR, 'rotM': M2},
                'L': {'bbox': bboxL, 'rotM': M1}}

    return full_images, side_patches, metadata, rotated_landmarks2,rotated_landmarks1, R, L,LL,RR


def read_pts_tibia_femur(pts_fname, spacing, padding):
    """
    Reads only the tibial and femoral plateaus' landmarks

    Parameters
    ----------
    pts_fname : str
        Full path to the file with points annotated with BoneFinder
    spacing : float
        Image spacing from DICOM file
    padding : int
        Padding to add to the image

    Returns
    -------
    out : tuple
        Femoral and tibial landmarks

    """
    points = np.round(read_pts(pts_fname) * 1. / spacing) + padding
    # landmarks_f = points[12:25, :]
    # landmarks_t = points[47:64, :]
    return points

def read_pts_tibia_femur2(pts_fname, spacing, padding):
    """
    Reads only the tibial and femoral plateaus' landmarks

    Parameters
    ----------
    pts_fname : str
        Full path to the file with points annotated with BoneFinder
    spacing : float
        Image spacing from DICOM file
    padding : int
        Padding to add to the image

    Returns
    -------
    out : tuple
        Femoral and tibial landmarks

    """
    points = np.round(read_pts(pts_fname) * 1. / spacing) + padding
    landmarks_f = points[12:25, :]
    landmarks_t = points[47:64, :]
    return landmarks_f, landmarks_t


def worker(job_data):
    """
    General worker to be used in multi-core processing of X-rays stored in DICOM
    format.

    Parameters
    ----------
    job_data : tuple or list
        Job data, which has sufficient information to extract
        filename to the DICOM image, full path to the landmarks (left and right ones),
        and, finally the path to save the results
    job_preproc_hook : function
        function, which must return scirpt's arguments, filename to the DICOM image, full path to the landmarks (left and right ones),
        and, finally the path to save the results

    Returns
    -------
    out : int
        Returns 0, if the data were successfully read and 1 if not.

    """
    args, fname, pts_fname_l, pts_fname_r, path_save = job_data
    if os.path.isfile(path_save):
        return 0
    res = read_dicom(fname)
    if res is None:
        return 1
    I, orig_spacing, img_metadata = res
    dd = img_metadata.pixel_array
    scale = orig_spacing / args.spacing

    I = process_xray(I).astype(np.uint8)
    I = pad(I, padding=args.pad)
    row, col = I.shape

    landmarks_l = read_pts_tibia_femur(pts_fname_l, orig_spacing, args.pad)
    landmarks_r = read_pts_tibia_femur(pts_fname_r, orig_spacing, args.pad)

    landmarks_r[:, 0] = col - landmarks_r[:, 0]


    # displ_img(I,landmarksTL, landmarksFL, landmarksTR, landmarksFR)
    landmarks2 = {'R': landmarks_r * scale,
                 'L': landmarks_l * scale}

    landmarks_fl, landmarks_tl = read_pts_tibia_femur2(pts_fname_l, orig_spacing, args.pad)
    landmarks_fr, landmarks_tr = read_pts_tibia_femur2(pts_fname_r, orig_spacing, args.pad)

    landmarks_fr[:, 0] = col - landmarks_fr[:, 0]
    landmarks_tr[:, 0] = col - landmarks_tr[:, 0]

    # displ_img(I,landmarksTL, landmarksFL, landmarksTR, landmarksFR)
    landmarks = {'TR': landmarks_tr * scale,
                 'FR': landmarks_fr * scale,
                 'TL': landmarks_tl * scale,
                 'FL': landmarks_fl * scale}



    # Scaling to the new spacing
    I = cv2.resize(I, (int(col * scale), int(row * scale)), interpolation=cv2.INTER_CUBIC)
    try:
        full_images, side_patches, metadata, rotated_landmarks2,rotated_landmarks1, R, L,LL,RR = extract_patches(I, landmarks, landmarks2, args.spacing, args.sizemm, args.o_sizemm_w, args.o_sizemm_h)
    except:
        return 1
    #import os
    imageR, new_spacing = resample(RR, img_metadata)
    imageL, new_spacing = resample(RR, img_metadata)





    import matplotlib
    import matplotlib.pyplot as plt
    # plt.figure(figsize=(10.8, 10.8))
    # plt.imshow(image, cmap=plt.cm.bone)
    # Displacement = metadata['R']['bbox']
    # rightlanmark = landmarks2['R']
    # rightlanmark[:, 0] -= Displacement[0]
    # rightlanmark[:, 1] -= Displacement[1]
    # plt.scatter(x=rotated_landmarks1[:, 0], y=rotated_landmarks1[:, 1], c='r', s=10)
    # plt.axis('off')
    os.mkdir(path_save)
    # plt.savefig(path_save + 'R.jpg', bbox_inches='tight')
    # plt.figure(figsize=(10.8, 10.8))
    # plt.imshow(L, cmap=plt.cm.bone)
    # plt.scatter(x=rotated_landmarks2[:, 0], y=rotated_landmarks2[:, 1], c='r', s=10)
    # plt.axis('off')
    # plt.savefig(path_save + 'L.jpg', bbox_inches='tight')

    #np.save(path_save, [full_images, side_patches, metadata, landmarks, landmarks2, img_metadata.Modality, args.spacing])
    R = R.astype('float')
    L = L.astype('float')
    matplotlib.image.imsave(path_save + 'R.jpg', imageR,
                            cmap=plt.cm.bone)

    matplotlib.image.imsave(path_save + 'L.jpg', imageL,
                            cmap=plt.cm.bone)


    return 0, R.shape[0], L.shape[0], L, R
