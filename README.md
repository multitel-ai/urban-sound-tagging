# UrbanNet [DCASE2020 Task 5](http://dcase.community/challenge2020/task-urban-sound-tagging-with-spatiotemporal-context)

This git contains code for CRNNs we used to achieve first rank in Task 5 of the DCASE 2020 challenge. This task focuses on hierarchical multilabel urban sound tagging with spatiotemporal context. The technical report can be found [here](https://arxiv.org/pdf/2008.10413.pdf).

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
pip install numpy scikit-learn pandas tqdm albumentations librosa tensorboard torch torchvision oyaml pytorch-lightning numba==0.49
~~~

## Editing `config.py`

You should edit PATH in `config.py` to match the directory in which everything will be stored.

## Data download and preprocessing

Use the following to download the dataset and precompute inputs for TALNet and CNN14.

WARNING : It requires about 30Go of free space.

~~~bash
python data_prep.py --download --mel
~~~

If download fails, you can rename/delete old sonycust folder and it will download it again.

If you want to manualy download and decompress files, you have to put everything in the `audio` directory. Then you have to use the aboce command without the `--download`. We use the version 2.2 of SONYCUST available [here](https://zenodo.org/record/3873076).

Your SONYC-UST folder should look like this :
~~~bash
.
└── SONC-UST                    # Given by path_to_SONYCUST in config.py
    ├── audio                   # Put all audio in it
    ├── model                   # Put TALNet weights here
    ├── annotation.csv          
    ├── best2.csv               # Our relabel file
    └── dcase-ust-taxonomy.yaml
~~~

## Training

The code should work on both CPU and GPU.
Add `--gpu` followed by the number of GPUs to use them.

Add `--seed` followed by the seed for to set seed for random operations.

~~~bash
python training_system1.py --gpu 1 --seed 1
~~~

You can add `--cleaning_strat Relabeled --relabeled_name INSERT_CSV_HERE` to train on a specific relabeled dataset. See below for relabeling.
The pretrained TALNet on Audioset can be found at http://islpc21.is.cs.cmu.edu/yunwang/git/cmu-thesis/model/TALNet.pt.

Note that system 2 and 3 recquires the pretrained TALNet model.

## Relabeling

Once system3 has been trained a first time, we can use it to relabel everything.

You have to specify the ckpt path and the `hparams.yml` path. Because of a bug in the pytorch-lighnting build used, `hparams.yml` has to be edited to remove the early stop part.

~~~bash
python relabel.py --path_to_ckpt INSERT_HERE --path_to_yaml INSERT_HERE
~~~

## Generating submission file

Like the relabeling part, hparams.yml has to be edited to remove the early stop part.

Once it is done, you have to specify both the path to the checkpoint of the model and the path to the hparams edited.

~~~bash
python sub_system1 --path_to_ckpt INSERT_HERE --path_to_yaml INSERT_HERE
~~~

## Citing

~~~bibtex
@article{Arnault2020,
  title={CRNNs for Urban Sound Tagging with spatiotemporal context},
  author={Arnault, Augustin and Riche, Nicolas},
  journal={arXiv preprint arXiv:2008.10413},
  year={2020}
}
~~~
