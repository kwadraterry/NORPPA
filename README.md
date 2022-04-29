# NORPPA: NOvel Ringed seal re-identification by Pelage Pattern Aggregation


## Introduction

This repository contains an implementation of **NORPPA** method for Saimaa ringed seal (*Pusa hispida saimensis*) re-identification. Access to large image volumes through camera trapping and crowdsourcing provides novel possibilities for animal monitoring and conservation and calls for automatic methods for analysis, in particular, when re-identifying individual animals from the images. The proposed method NOvel Ringed seal re-identification by Pelage Pattern Aggregation (**NORPPA**) utilizes the permanent and unique pelage pattern of Saimaa ringed seals and content-based image retrieval techniques. First, the query image is preprocessed, and each seal instance is segmented. Next, the seal’s pelage pattern is extracted using a U-net encoder-decoder based method. Then, CNN-based affine invariant features are embedded and aggregated into Fisher Vectors. Finally, the cosine distance between the Fisher Vectors is used to find the best match from a database of known individuals. We perform extensive experiments of various modifications of the method on a new challenging Saimaa ringed seals re-identification dataset. 

We are making the **NORPPA** code available to the research community free of charge. If you use this code in your research, we kindly ask that you reference our papers listed below:
 

Ekaterina Nepovinnykh, Ilia Chelak, Tuomas Eerola, and Heikki Kälviäinen, "NORPPA: NOvel Ringed seal re-identification by Pelage Pattern Aggregation"

Ekaterina Nepovinnykh, Tuomas Eerola, Vincent Biard, Piia Mutka, Marja Niemi, Mervi Kunnasranta, and Heikki Kälviäinen, "SealID: Saimaa ringed seal re-identification database"

## Setup
This code has been tested and run on OS Ubuntu. 

The environment for running this code can set up using `conda` by running the following script:

```
. ./setup.sh
```

In order to use tonemapping the `pfstmo` package should be installed by running:

```
apt-get install pfstmo
```

## Usage

The code can be tested using a SealID dataset available at https://doi.org/10.23729/0f4a3296-3b10-40c8-9ad3-0cf00a5a4a53 

In order to download a dataset get a single use download link from https://etsin.fairdata.fi/dataset/22b5191e-f24b-4457-93d3-95797c900fc0/data

You will only need `full images.zip` for the reidentification, generate a link to that.

An example of using the provided code is given in `notebook_test.ipynb`. 

## Content

### Tonemapping

Tone-mapping approach equalizes the contrast in dark and bright image regions to make the seal fur pattern more clear. Tonemapping algorithm from [`pfstmo` framework](http://pfstools.sourceforge.net/pfstmo.html) is used.

### Segmentation
Segmentationis used to extract the seal from the background. [Detectron2 framework](https://github.com/facebookresearch/detectron2) is used.

### Pattern extraction
Pattern extraction refers to a segmentation of unique pelage pattern from a seal. 

### Re-identification
Task for the re-identification algorithm is to search for the best match from the database for the given query image.

First, patches (small regions) are extracted from pattern images using [HesAffNet](https://github.com/ducha-aiki/affnet) and encoded using [HardNet](https://github.com/DagnyT/hardnet). 

Next, Fisher Vectors are created using [Cyvealfeat framework](https://github.com/menpo/cyvlfeat)



## Copyright notice

The NORPPA code is Copyright © 2022 by Computer Vision and Pattern Recognition Laboratory. The code can be freely downloaded and used for non-profitable scientific purposes. Proper citing of this resource is expected if the code is used in research or other reporting. The code is meant to be useful, but it is distributed WITHOUT ANY WARRANTY, and without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

 
Inclusion of this code or even parts of it in a proprietary program is not allowed without a written permission from the owners of the copyright. If you wish to obtain such a permission, you should contact

 

Computer Vision and Pattern Recognition Laboratory  
LUT University  
PO Box 20  
FI-53851 Lappeenranta  
FINLAND

 

If you find any errors or bugs, or have comments related to the code contents, please contact the authors by emai: ekaterina.nepovinnykh@lut.fi

 

## Acknowledgements

The research was carried out in the CoExist project (Project ID: KS1549) funded by the European Union, the Russian Federation and the Republic of Finland via The South-East Finland -- Russia CBC 2014-2020 programme.




