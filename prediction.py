import json
import time
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models.matching import OptimalMatching
from models.backbone import R2U_Net, NonMaxSuppression, DetectionBranch

from dataloader_crowdai import CrowdAI
from pycocotools.coco import COCO


def bounding_box_from_points(points):
    points = np.array(points).flatten()
    even_locations = np.arange(points.shape[0]/2) * 2
    odd_locations = even_locations + 1
    X = np.take(points, even_locations.tolist())
    Y = np.take(points, odd_locations.tolist())
    bbox = [X.min(), Y.min(), X.max()-X.min(), Y.max()-Y.min()]
    bbox = [int(b) for b in bbox]
    return bbox


def single_annotation(image_id, poly):
    _result = {}
    _result["image_id"] = int(image_id)
    _result["category_id"] = 100 
    _result["score"] = 1
    _result["segmentation"] = poly
    _result["bbox"] = bounding_box_from_points(_result["segmentation"])
    return _result


def prediction(batch_size, images_directory, annotations_path):

    # Load network modules
    model = R2U_Net()
    model = model.cuda()
    model = model.train()

    head_ver = DetectionBranch()
    head_ver = head_ver.cuda()
    head_ver = head_ver.train()

    suppression = NonMaxSuppression()
    suppression = suppression.cuda()

    matching = OptimalMatching()
    matching = matching.cuda()
    matching = matching.train()

    # NOTE: The modules are set to .train() mode during inference to make sure that the BatchNorm layers 
    # rely on batch statistics rather than the mean and variance estimated during training. 
    # Experimentally, using batch stats makes the network perform better during inference.

    print("Loading pretrained model")
    model.load_state_dict(torch.load("./trained_weights/polyworld_backbone"))
    head_ver.load_state_dict(torch.load("./trained_weights/polyworld_seg_head"))
    matching.load_state_dict(torch.load("./trained_weights/polyworld_matching"))

    # Initiate the dataloader
    CrowdAI_dataset = CrowdAI(images_directory=images_directory, annotations_path=annotations_path)
    dataloader = DataLoader(CrowdAI_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)

    train_iterator = tqdm(dataloader)

    speed = []
    predictions = []
    for i_batch, sample_batched in enumerate(train_iterator):

        rgb = sample_batched['image'].cuda().float()
        idx = sample_batched['image_idx']

        t0 = time.time()

        features = model(rgb)
        occupancy_grid = head_ver(features)

        _, graph_pressed = suppression(occupancy_grid)

        poly = matching.predict(rgb, features, graph_pressed) 

        speed.append(time.time() - t0)


        for i, pp in enumerate(poly):
            for p in pp:
                predictions.append(single_annotation(idx[i], [p]))

        del features
        del occupancy_grid
        del graph_pressed
        del poly
        del rgb
        del idx

    print("Average model speed: ", np.mean(speed) / batch_size, " [s / image]")

    fp = open("predictions.json", "w")
    fp.write(json.dumps(predictions))
    fp.close()


if __name__ == '__main__':
    prediction(batch_size=6,
            images_directory="/home/stefano/Workspace/data/mapping_challenge_dataset/raw/val/images/",
            annotations_path="/home/stefano/Workspace/data/mapping_challenge_dataset/raw/val/annotation.json")
