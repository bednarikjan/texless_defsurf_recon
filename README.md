# Learning to Reconstruct Texture-less Deformable Surfaces from a Single View

Source code for the paper [Learning to Reconstruct Texture-less Deformable 
Surfaces from a Single View](https://arxiv.org/abs/1803.08908) presented at 
3DV 2018.

## Dependencies
Tested with `Python 3.6` and the following libraries:

- Keras 2.2.4
- tensorflow-gpu 1.12.0 
- numpy 1.16.1
- matplotlib 3.0.2
- PyYAML 3.13

## Dataset
The real-world dataset which was captured for the purposes of the paper is
available [here](https://cvlab.epfl.ch/data/texless-defsurf-data/). 
    
## How to run
### Training
`train.py` is the main entry point. It can be used in three modes:
1. Training from scratch:
    - `python train.py --config path_to_config_file`
2. Continue an (interrupted) train run by loading the model and optimizer state:
    - `python train.py --cont path_to_train_run_dir `
3. Train a new model which is initialized with weights from another one:
    - `python train.py --config path_to_config_file --model_state path_to_model_weights`
    
Each training run cerates a directory into which the configuration file is 
copied and model weights and optimizer parameters are being continuously saved.
    
### Configuration
All the parameters, namely the architecture, input data, optimizer settings,
learning rate scheduler etc. can be set using the configuration file `*.yaml`.
See `config_n.yaml` and `config_nd.yaml` for examples.
    
### Example of reproducing the `cloth-cloth` experiment
1. Set the parameters `path_imgs`, `path_normals` and `path_dmaps` to reflect the
path to the dataset.
2. Set the parameter `path_train_run` to a directory which will contain
training run data.
3. Train an architecture with normals stream only:
    - `python train.py --config config_n.yaml`
4. Train an architecture with normals and depth streams and initialize it with
the weights from the previous run.
    - `python train.py --config config_nd.yaml --model_state ../train_runs/N_wn1.0_k10.0/model_params_epN.h5` 
(N has to be replaced by an actual number)

