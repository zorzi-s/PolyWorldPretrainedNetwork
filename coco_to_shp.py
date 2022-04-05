from pycocotools.coco import COCO
import numpy as np
import json
from tqdm import tqdm
import shapefile


def cocojson_to_shapefiles(input_json, gti_annotations, output_folder):

    submission_file = json.loads(open(input_json).read())
    coco = COCO(gti_annotations)
    coco = coco.loadRes(submission_file)

    image_ids = coco.getImgIds(catIds=coco.getCatIds())

    for image_id in tqdm(image_ids):

        img = coco.loadImgs(image_id)[0]

        annotation_ids = coco.getAnnIds(imgIds=img['id'])
        annotations = coco.loadAnns(annotation_ids)

        list_poly = []
        for _idx, annotation in enumerate(annotations):
            poly = annotation['segmentation'][0]
            poly = np.array(poly)
            poly = poly.reshape((-1,2))

            poly[:,1] = -poly[:,1]
            list_poly.append(poly.tolist())

        number_str = str(image_id).zfill(12)
        w = shapefile.Writer(output_folder + '%s.shp' % number_str)
        w.field('name', 'C')
        w.poly(list_poly)
        w.record("polygon")
        w.close()

    print("Done!")


if __name__ == "__main__":
    cocojson_to_shapefiles(input_json="./predictions.json",
                            gti_annotations="/home/stefano/Workspace/data/mapping_challenge_dataset/raw/val/annotation.json",
                            output_folder="./shapefiles/")
