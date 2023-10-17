#!/bin/bash -l

usage()
{
    echo "Help:"
    echo "--conda - use conda instead of mamba."
    echo "--name - environment name to use."
}

env_manager=mamba
if ! command -v $env_manager &> /dev/null
then
    echo "mamba could not be found, defaulting back to conda"
    env_manager=conda
fi

env_name="norppa"

while [ "$1" != "" ]; do
    case $1 in
        --conda )                   shift
                                    env_manager=conda
                                    ;;
        --name )                    shift
                                    env_name=$1
                                    shift
                                    ;;
        -h | --help )               usage
                                    exit
                                    ;;
        * )                         break
    esac
    shift
done

source "$(dirname $(which $env_manager))/activate"

prefix="/$USER/env/$env_name"


echo "Creating and activating $env_name environment"
$env_manager create -y --prefix $prefix python=3.8 cudatoolkit=11.1 cyvlfeat opencv ffmpeg cudnn -c conda-forge # tensorflow=2.6

# echo "Current python: $(which python)"

run_in_env="$env_manager run -p $prefix --no-capture-output"

echo "Installing pip requirements"
$run_in_env python -m pip install torch==2.0.1 torchvision tensorflow==2.8
echo "Installing requirements file"
$run_in_env python -m pip install -r ./requirements.txt

$run_in_env python -m pip install typing-extensions kornia_moons --upgrade

$run_in_env python -m pip install ipykernel
$run_in_env python -m ipykernel install --user --name=$env_name