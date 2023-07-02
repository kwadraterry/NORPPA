#!/bin/bash

usage()
{
    echo "Help:"
    echo "--cpu - use cpu only packages."
}

cpu=false

while [ "$1" != "" ]; do
    case $1 in
        --cpu )                     shift
                                    cpu=true
                                    ;;
        -h | --help )               usage
                                    exit
                                    ;;
        * )                         break
    esac
    shift
done

echo "Creating and activating norppa environment"
conda create -y --prefix /$USER/env/norppa python=3.7 cudatoolkit=11.1 cyvlfeat opencv ffmpeg cudnn -c conda-forge
conda activate /$USER/env/norppa

echo "Installing ipykernel, cyvlfeat and opencv"
# conda install -y cyvlfeat libopencv opencv py-opencv -c conda-forge

# echo "Installing pytorch, torchvision and detectron"
if [ "$cpu" = true ]
then
  # conda install -y pytorch=1.10 torchvision cpuonly cyvlfeat libopencv opencv py-opencv -c conda-forge
  pip install torch==1.10.1+cpu torchvision==0.11.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
  python -m pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
else
  # conda install -y pytorch=1.10 torchvision cyvlfeat libopencv opencv py-opencv -c conda-forge
  python -m pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
  python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
fi

# conda install -y ipykernel cyvlfeat opencv tensorflow-gpu -c conda-forge

# conda install kornia -c conda-forge

echo "Installing pip requirements"
pip3 install -r ./requirements.txt

conda update -y ffmpeg
#echo "Installing detectron2"
#conda install -c conda-forge detectron2
# if [ "$cpu" = true ]
# then
#   python3 -m pip install detectron2 -f \
#     https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.5/index.html
# else
#   python3 -m pip install detectron2 -f \
#     https://dl.fbaipublicfiles.com/detectron2/wheels/gpu/torch1.5/index.html
# fi

#echo "Installing tensorflow"
