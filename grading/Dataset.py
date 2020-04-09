
import cv2


class OAIdataset():
    """datasetA."""

    def __init__(self, csv_file, root_dir, transform):
        """
        Args:
            csv_file (string): Path to the csv file with KL grade and ID.
            root_dir (string): Directory with all the images.
        """
        self.landmarks_frame = csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        Input = self.landmarks_frame.loc[idx]

        imageID = Input['ParticipantID']
        landmarks = Input['Label']
        side = str(Input['side'])
        id = str(imageID)

        file = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/test 1/patches/adaptive/' +\
               str(landmarks) + '/' + id + '_' + side +'.png'
# '_' + str(Input['SeriesDescription'])
        from PIL import Image
        image = Image.open(file)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'landmarks': landmarks, 'imageID': imageID}

        return sample
