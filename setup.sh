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
# $env_manager activate /$USER/env/$env_name

# echo "Current python: $(which python)"

run_in_env="$env_manager run -p /$USER/env/$env_name"

echo "Installing pip requirements"
$run_in_env python -m pip install torch torchvision tensorflow==2.8
echo "Installing requirements file"
$run_in_env python -m pip install -r ./requirements.txt

$run_in_env python -m pip install typing-extensions kornia_moons --upgrade

$run_in_env python -m pip install ipykernel
$run_in_env python -m ipykernel install --user --name=$env_name