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

env_name="norppa_test"

echo "Creating and activating $env_name environment"
conda create -y --prefix /$USER/env/$env_name python=3.7 cudatoolkit=11.1 cyvlfeat opencv ffmpeg cudnn tensorflow==2.6 -c conda-forge
conda init bash
conda activate /$USER/env/$env_name

# echo "Installing ipykernel, cyvlfeat and opencv"
# conda install -y cyvlfeat libopencv opencv py-opencv -c conda-forge

echo "Installing pytorch, torchvision and detectron"
if [ "$cpu" = true ]
then
  # conda install -y pytorch=1.10 torchvision cpuonly cyvlfeat libopencv opencv py-opencv -c conda-forge
  python -m pip install torch==1.10.1+cpu torchvision==0.11.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
  python -m pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
else
  # conda install -y pytorch=1.10 torchvision cyvlfeat libopencv opencv py-opencv -c conda-forge
  python -m pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
  python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
fi

echo "Installing pip requirements"
python -m pip install -r ./requirements.txt

# mkdir -p $CONDA_PREFIX/etc/conda/activate.d
# echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# conda update -y ffmpeg

# python -m pip install --upgrade numpy

pip install typing-extensions kornia_moons --upgrade

# python -m pip install ipykernel
python -m ipykernel install --user --name=$env_name