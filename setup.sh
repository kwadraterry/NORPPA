#!/bin/bash -l

usage()
{
    echo "Help:"
    echo "--cpu - use cpu only packages."
}

env_manager=mamba

while [ "$1" != "" ]; do
    case $1 in
        --conda )                     shift
                                    env_manager=conda
                                    ;;
        -h | --help )               usage
                                    exit
                                    ;;
        * )                         break
    esac
    shift
done

source "$(dirname $(which $env_manager))/activate"
env_name="norppa"



echo "Creating and activating $env_name environment"
$env_manager create -y --prefix /$USER/env/$env_name python=3.8 cudatoolkit=11.1 cyvlfeat opencv ffmpeg cudnn -c conda-forge # tensorflow=2.6
# $env_manager init bash
# . /root/mambaforge/bin/activate
$env_manager activate /$USER/env/$env_name

echo "Current python: $(which python)"

echo "Installing pip requirements"
python -m pip install torch torchvision tensorflow==2.8
python -m pip install -r ./requirements.txt

pip install typing-extensions kornia_moons --upgrade

python -m pip install ipykernel
python -m ipykernel install --user --name=$env_name