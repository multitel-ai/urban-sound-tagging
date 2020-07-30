# UrbanNet [DCASE2020 Task 5](http://dcase.community/challenge2020/task-urban-sound-tagging-with-spatiotemporal-context)

This git contains code for CRNNs we used to achieve first rank in Task 5 of the DCASE 2020 challenge. This task focuses on hierarchical multilabel urban sound tagging with spatiotemporal context. The technical report can be found [here](http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Arnault_70_t5.pdf).

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

If you want to manualy download and decompress files, you have to put everything in the `audio` directory. Then you have to use the aboce command without the `--download`.

## Training

The code should work on both CPU and GPU.
Add `--gpu` followed by the number of GPUs to use them.

Add `--seed` followed by the seed for to set seed for random operations.

~~~bash
python training_system1 --gpu 1 --seed 1
~~~

You can add `--cleaning_strat Relabeled --relabeled_name INSERT_CSV_HERE` to train on a specific relabeled dataset. See below for relabelling.
The pretrained TALNet on Audioset can be found at http://islpc21.is.cs.cmu.edu/yunwang/git/cmu-thesis/model/TALNet.pt.

## Relabelling

Once system3 has been trained a first time, we can use it to relabel everything.

You have to specify the ckpt path and the `hparams.yml` path. Because of a bug in the pytorch-lighnting build used, `hparams.yml` has to be edited to remove the early stop part.

~~~bash
python relabel.py --path_to_ckpt INSERT_HERE --path_to_yaml INSERT_HERE
~~~

## Generating submission file

Like the relabelling part, hparams.yml has to be edited to remove the early stop part.

Once it is done, you have to specify both the path to the checkpoint of the model and the path to the hparams edited.

~~~bash
python sub_system1 --path_to_ckpt INSERT_HERE --path_to_yaml INSERT_HERE
~~~

## Citing

~~~bibtex
@techreport{Arnault2020,
    Author = "Arnault, Augustin and Riche, Nicolas",
    title = "{CRNNs} for Urban Sound Tagging with Spatiotemporal Context",
    institution = "DCASE2020 Challenge",
    year = "2020",
    month = "October",
    abstract = "This paper describes CRNNs we used to participate in Task 5 of the DCASE 2020 challenge. This task focuses on hierarchical multilabel urban sound tagging with spatiotemporal context. The code is available to our GitHub repository at https://github.com/multitel-ai/urban-sound-tagging."
}
~~~
