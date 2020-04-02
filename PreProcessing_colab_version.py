import pandas as pd
from Execution_Segmentation import Execution_Segmentation_amir
import png
import numpy as np

List_all = pd.read_csv('C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/ALL.csv')
sizesoa = []
sizesnooa = []

for idx in range(len(List_all)):

    listname = List_all.loc[idx]

    roi_med_t, roi_lat_t = Execution_Segmentation_amir(listname)

    imageID = listname['ParticipantID']
    landmarks = listname['Label']
    Side = listname['side']

    address = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/patches/'

    if listname['side'] == 'med':

        png.from_array(roi_med_t, mode="L")\
            .save(address + str(landmarks) + '/' + str(imageID) + '_' + str(Side) + '.png')

        sizesoa.append(len(roi_med_t))

    else:
        png.from_array(roi_lat_t, mode="L") \
            .save(address + str(landmarks) + '/' + str(imageID) + '_' + str(Side) + '.png')

        sizesnooa.append(len(roi_med_t))



