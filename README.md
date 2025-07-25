
<img src="assets/logo.png" width="350">


### Research @ TUGraz & BlackShark.ai (CVPR 2022)

# PolyWorld Inference and Evaluation Code

PolyWorld is a research project conducted by the Institute of Computer Graphics and Vision of TUGraz, in collaboration with BlackShark.ai. PolyWorld is a neural network that extracts polygonal objects from an image in an end-to-end fashion. The model detects vertex candidates and predicts the connection strenght between each pair of vertices using a Graph Neural Network. This repo includes inference code and pretrained weights for evaluating PolyWorld on the CrowdAI Mapping Challenge dataset.

<p align="center">
  <img src="assets/teaser.png" width="450">
</p>

- Paper PDF: [PolyWorld: Polygonal Building Extraction with Graph Neural Networks in Satellite Images](https://arxiv.org/abs/2111.15491)

- Authors: Stefano Zorzi, Shabab Bazrafkan, Stefan Habenschuss, Friedrich Fraundorfer

- Video: [YouTube link](https://youtu.be/C80dojBosLQ)

- Poster: [GDrive link](https://drive.google.com/file/d/1S1i6762r5kEaIe5GVpgTluEC8CfUzqJ0/view?usp=sharing)



## Dependencies

- pycocotools
- pyshp
- torch

## Getting started

After cloning the repo, download the _polyworld_backbone_ pre-trained weights from [here](https://drive.google.com/file/d/1pK7FDg6CphCvSiOyeC2Lx3UEOQ2UPrdj/view?usp=sharing), and place the file in the _trained_weights_ folder.

The CrowdAI Mapping Challenge dataset can be downloaded [here](https://drive.google.com/drive/folders/1pm0wlp0gmSMDgCcDOwl8VGjS9nXfvYUd?usp=sharing).

## Run the evaluation on the CrowdAI Mapping Challenge dataset

To run the evaluation, specify batch size, image folder, and annotation file of the CrowdAI dataset in the main function of the _prediction.py_ script.
Then simply run:

`python prediction.py`

The code is tested on an Nvidia RTX 3090 using _batch_size = 6_.

During inference, the script converts the predicted polygons to coco format and saves a json file (_predictions.json_).

If you wish to visualize the results in QGIS, we suggest to convert the predictions from coco json format to shapefile using the _coco_to_shp.py_ script. 
To run the conversion, specify the json file and the output folder in the main function, and then type:

`python coco_to_shp.py`

In order to compute AP and AR metrics with the COCO API, or the MTA metric, please use the script provided by the [Frame Field Learning](https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning) repo.

If you want to compute IoU and C-IoU metrics, use the _coco_IoU_cIoU.py_ script. To run the evaluation, specify json file and ground truth annotations in the main function and then run:

`python coco_IoU_cIoU.py`
 
## Download results

A download link for the PolyWorld predictions corresponding to the val-set of the CrowdAI dataset is also provided:

- [json results](https://drive.google.com/drive/folders/1Btau6_4y04GfNTC1uY_c_UTzH4D5k8D4?usp=sharing): here you can find the output annotations in json format with and without using refinement vertex offsets.

- [shp results](https://drive.google.com/drive/folders/1_AsNVYv-el-yu_ib-54PkYRMQSkz1PJe?usp=sharing): here you can find archives containing the shapefile annotations ready to be visualized in QGIS.

## BibTeX citation

If you use any ideas from the paper or code from this repo, please consider citing:

```
@inproceedings{zorzi2022polyworld,
  title={PolyWorld: Polygonal Building Extraction with Graph Neural Networks in Satellite Images},
  author={Zorzi, Stefano and Bazrafkan, Shabab and Habenschuss, Stefan and Fraundorfer, Friedrich},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1848--1857},
  year={2022}
}

@inproceedings{zorzi2023re,
  title={Re: PolyWorld-A graph neural network for polygonal scene parsing},
  author={Zorzi, Stefano and Fraundorfer, Friedrich},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16762--16771},
  year={2023}
}
```
