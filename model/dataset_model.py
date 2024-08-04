import torch
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import random

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, annot_df, transform=None):
        """Initialize the CustomDataset.
        Args:
            annot_df (DataFrame): DataFrame containing image annotations.
            transform (callable, optional): Optional transform to be applied on a sample."""
        self.annot_df = annot_df
        self.root_dir = ""  # Root directory of images
        self.transform = transform
    def __len__(self):
        # return length (numer of rows) of the dataframe
        return len(self.annot_df)
    def __getitem__(self, idx):
        # use image path column (index = 1) in csv file
        image_path = self.annot_df.iloc[idx, 1]
        image = cv2.imread(image_path)  # read image by cv2
        # convert from BGR to RGB for matplotlib
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # use class name column (index = 2) in csv file
        class_name = self.annot_df.iloc[idx, 2]
        # use class index column (index = 3) in csv file
        class_index = self.annot_df.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        # when accessing an instance via index, 3 outputs are returned - the image, class name and class index
        return image, class_name, class_index
    def visualize(self, number_of_img=10, output_width=12, output_height=6):
        
        """Visualize random samples from the dataset.
        Args:
            number_of_img (int, optional): Number of images to visualize.
            output_width (int, optional): Width of the output figure.
            output_height (int, optional): Height of the output figure."""
        plt.figure(figsize=(output_width, output_height))
        for i in range(number_of_img):
            idx = random.randint(0, len(self.annot_df))  # Select a random index
            image, class_name, class_index = self.__getitem__(idx)  # Get sample
            ax = plt.subplot(2, 5, i+1)  # Create subplot
            ax.title.set_text(class_name + '-' + str(class_index))  # Set title
            if self.transform == None:
                plt.imshow(image)  # Show image
            else:
                plt.imshow(image.permute(1, 2, 0))  # Show transformed image


def create_validation_dataset(dataset, validation_proportion):
    """
    Split a dataset into training and validation sets.

    Args:
        dataset (Dataset): Dataset to split.
        validation_proportion (float): Proportion of dataset to use for validation.

    Returns:
        Dataset: Training dataset.
        Dataset: Validation dataset.
    """
    if (validation_proportion > 1) or (validation_proportion < 0):
        return "The proportion of the validation set must be between 0 and 1"
    else:
        dataset_size = int((1 - validation_proportion) * len(dataset))  # Calculate dataset size
        validation_size = len(dataset) - dataset_size  # Calculate validation set size
        dataset, validation_set = torch.utils.data.random_split(
            dataset, [dataset_size, validation_size])  # Split dataset
        return dataset, validation_set