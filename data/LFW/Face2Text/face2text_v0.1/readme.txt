Face2Text data readme

-----
Version
-----
0.1

-----
Release date
-----
16/May/2018

-----
Contents
-----
readme.txt: Information about this dataset.
raw.json: Dataset with all descriptions included and left unaltered as written by participants.
clean.json: Dataset with filtered and cleaned descriptions.

-----
Paper
-----
Face2Text: Collecting an Annotated Image Description Corpus for the Generation of Rich Face Descriptions (http://www.lrec-conf.org/proceedings/lrec2018/summaries/226.html)

-----
Facebook
-----
https://www.facebook.com/Research-in-Vision-and-Language-group-RiVaL-114101215851716/

-----
Website
-----
http://rival.research.um.edu.mt/

-----
Stats
-----
Number of images: 400
Number of descriptions: 1237
Maximum number of descriptions per image: 4
Minimum number of descriptions per image: 1
Maximum number of tokens in descriptions: 94
Minimum number of tokens in descriptions: 2
Characters used in descriptions: \n !"$%'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWYZ`abcdefghijklmnopqrstuvwxyzèé

Note that '\n' means 'new line'.

-----
Description
-----
Face2Text is an ongoing project to collect a dataset of natural language descriptions of human faces. 400 randomly selected images were used from the Labelled Faces in the Wild dataset (http://vis-www.cs.umass.edu/lfw/) and put on a website where people were asked to write descriptions of each face. See paper for more information.

The dataset is provided in json format structured as a list of objects consisiting of the following:
    img_id: A unique id referring to an image of a face.
    image: The directory of the image file.
    descriptions: A list of objects consisting of:
        desc_id: A unique description id
        text: A string description of the image.

Here is an example:
    [
        "img_id": 198,
        "image": "Adam_Ant/Adam_Ant_0001.bmp",
        "descriptions": [
            {
                "desc_id": 979,
                "text": "big face yellow hair, big ears long nose"
            },
            ...
        ],
        ...
    ]

The image directory refers to a face photo in the Labelled Faces in the Wild dataset (http://vis-www.cs.umass.edu/lfw/lfw.tgz). We do not include images here; you will need to download them and select them separately.

A cleaned version of the dataset has been provided. Only minor superficial changes have been made but we intend to perform a more thorough cleaning in future versions. In order to filter some of the garbage submissions, only descriptions that contain a space were included in the dataset. Spaces at the beginning or end of a description were removed. Spaces before or after a new line where removed. Multiple consecutive spaces were replaced with a single space. Non-English descriptions were removed.

-----
License
-----
This data is work in progress and can only be used for research and/or teaching purposes. It is being distributed by the RiVaL Group, University of Malta in its current state on condition that its users explicitly acknowledge the source, such as by citing our paper (bibtex in the link above), and do not distribute the data further without the express permission of the RiVaL Group.
