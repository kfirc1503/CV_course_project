"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """

    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""
        if index >= len(self.real_image_names):
            label = 1
            image_name = self.fake_image_names[index - len(self.real_image_names)]
        else:
            label = 0
            image_name = self.real_image_names[index]
        image_category = 'fake' if label == 1 else 'real'
        img_path = os.path.join(self.root_path, image_category, image_name)
        img = Image.open(img_path, 'r')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        """Return the number of images in the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""
        total_img_num = len(self.real_image_names) + len(self.fake_image_names)
        return total_img_num
