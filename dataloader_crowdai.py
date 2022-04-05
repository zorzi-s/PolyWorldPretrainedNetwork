import numpy as np
import random
from skimage import io
from skimage.transform import resize
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CrowdAI(Dataset):
    """CrowdAI dataset"""

    def __init__(self, images_directory, annotations_path):

        self.IMAGES_DIRECTORY = images_directory
        self.ANNOTATIONS_PATH = annotations_path

        self.coco = COCO(self.ANNOTATIONS_PATH)

        self.image_ids = self.coco.getImgIds(catIds=self.coco.getCatIds())

        self.len = len(self.image_ids)

        self.window_size = 320
        self.max_points = 256


    def loadSample(self, idx):

        idx = self.image_ids[idx]

        img = self.coco.loadImgs(idx)[0]
        image_path = self.IMAGES_DIRECTORY + img['file_name']

        image = io.imread(image_path)
        image = resize(image, (self.window_size, self.window_size, 3), anti_aliasing=True, preserve_range=True)

        annotation_ids = self.coco.getAnnIds(imgIds=img['id'])
        annotations = self.coco.loadAnns(annotation_ids)
        random.shuffle(annotations)

        image_idx = torch.tensor([idx])
        image = torch.from_numpy(image)
        image = image.permute(2,0,1) / 255.0

        sample = {'image': image, 'image_idx': image_idx}
        return sample


    def __len__(self):
        return self.len


    def __getitem__(self, idx):

        sample = self.loadSample(idx)
        return sample


