# UrbanNet [DCASE2020 Task 5](http://dcase.community/challenge2020/task-urban-sound-tagging-with-spatiotemporal-context)

This git contains code for CRNNs we used to participate in Task 5 of the DCASE 2020 challenge. This task focuses on hierarchical multilabel urban sound tagging with spatiotemporal context. If you want more information, our paper is [here](https://www.overleaf.com/project/5ee0b58cc34d6a0001694eb3).

## Environement setup

Python version recquired : 3.6 (Higher might work).
We recommand first to create a new environment in conda/virtualenv then to activate it.

Pip install

~~~bash
pip install -r requirements.txt
~~~

Conda install

~~~bash
conda env create -f environment.yml
~~~

Manual install

~~~bash
pip install numpy scikit-learn pandas tqdm albumentations librosa tensorboard torch torchvision oyaml pytorch-lightning
~~~

## Editing `config.py`

You should edit PATH in `config.py` to match the directory in which everything will be stored.

## Data download and preprocessing

Use the following to download the dataset and precompute inputs for TALNet and CNN14.

WARNING : It requires about 40Go of free space.

~~~bash
python data_prep.py --download --mel --resample
~~~

## Training

The code should work on both CPU and GPU.
Add `--gpu` followed by the number of GPUs to use them.

Add `--seed` followed by the seed for to set seed for random operations.

~~~bash
python training_system1 --gpu 1 --seed 1
~~~

## Relabelling

Once system3 has been trained a first time, we can use it to relabel everything.

~~~bash
python relabel.py
~~~

## Generating submission file

Because of a bug in the pytorch-lighnting build used, hparams.yml has to be edited to remove the early stop part.
Once it is done, you have to specify both the path to the checkpoint of the model and the path to the hparams edited.

~~~bash
python sub_system1
~~~