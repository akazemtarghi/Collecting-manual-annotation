
import cv2
import pandas as pd
import numpy as np
class OAIdataset():
    """datasetA."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with KL grade and ID.
            root_dir (string): Directory with all the images.
        """
        self.landmarks_frame = csv_file
        #self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        Input = self.landmarks_frame.loc[idx]
        imageID = Input['ParticipantID']
        landmarks = Input['Label']
        side = str(Input['side'])
        #slice = str(Input['slice'])
        id = str(imageID)

        file = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/patches/standard/' +\
               str(landmarks) + '/' + id  + '_' + side +'_'+ str(Input['SeriesDescription']) + '.png'
# '_' + str(Input['SeriesDescription'])

        # image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        from PIL import Image
        image = Image.open(file)
        # image = np.load(file)
        # image = np.uint8(image)


        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'landmarks': landmarks, 'imageID': imageID}

        return sample
