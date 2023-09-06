<h1>PDiscoNet: Semantically consistent part discovery for fine-grained recognition</h1> 
First, download the datasets here:

1. CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. Use the Google Drive link, download img/img_celeba.7z, and unzip. 
2. CUB: https://data.caltech.edu/records/65de6-vp158. Get CUB_200_2011.tgz.
3. PartImageNet: https://github.com/TACJu/PartImageNet. Download the PartImageNet_OOD version.

For the default folder structure, clone the git repo and extract the datasets into their respective folders in ```/datasets/```s (extract the CelebA dataset in a new folder ```/celeba/unaligned/```). So, the default folder structure is:
```
├── celeba
├── cub
├── datasets
│   ├── celeba
│   │	└── unaligned
│   ├── cub
│   │   ├── CUB_200_2011
│   └── partimagenet
│       ├── test
│       ├── train
│       └── val
└── partimagenet
```
Next, build the conda environment using the environment.yml file. 

Finally, there are two more steps to prepare the datasets:
<h4>PartImageNet</h4>
Finally, to prepare the PartImageNet dataset, there is one more step. When we first downloaded the datasets, the PartImageNet_OOD version of the data was the only existing version. However, in this version, the sets of classes in 'test', 'train', and 'val' were disjoint. Thus, we created two subsets of the 'train' dataset: a training subset and a testing subset. Simply run prep.py in datasets/partimagenet to prepare the dataset.

<h2>Training and testing the model</h2>
The argument parser has the following parameters:

```--model_name``` determines the name under which the file containing the parameters will be saved, and the folder in which the results will be saved. The parameters file will be saved as ```./[model_name].pt```, and the results will be saved in ```../results_[model_name]```.

```--data_path``` is the folder containing the dataset. For the CelebA dataset it's the folder containing ```unaligned/```. For CUB it's the ```CUB_200_2011``` folder, and for PartImageNet it's the folder containing ```train/```, ```test/```, and ```val/```.

```--num_parts``` is the number of parts the model should use. In the paper we trained the model with 4, 8, and 16 parts for CelebA and CUB, and 8, 25, and 50 parts for PartImageNet.

```--lr``` is the learning rate. We used ```1e-4``` for all datasets.

```--batch_size``` is the batch size, we used 15 for CUB, and 20 for CelebA and PartImageNet.

```--image_size``` is the resolution to which the input images will be cropped. First, the short edge of the raw image is resized to ```image_size```, and then the resized image is cropped to ```(image_size x image_size)```.

```--epochs``` is the number of epochs to run. The default for CelebA is 15, the default for CUB is 28, and the default for PartImageNet is 20.

```--pretrained_model_name``` is only used when evaluating the model, or when continuing the training process with a previously saved model. It should be equal to the ```--model_name``` parameter used when the model was trained. Used in conjunction with ```--warm_start True```.

```--save_maps``` determines whether attention maps are saved in the ```validation``` function in ```main.py```.

```--warm_start``` is used in conjunction with ```---pretrained_model_name```: when you use ```--warm_start True```, you should also specify a model name in ```--pretrained_model_name```.


```--only_test``` is used when you do not want to train the model, and instead you only wish to evaluate using a set of parameters. When you use this option, you should also use ```--warm_start``` and ```--pretrained_model_name```.

See below for some examples for each dataset.

<h3>CelebA</h3>
<h5>Training</h5>

```
python main.py --model_name celeb_8parts --data_path ../datasets/celeba --num_parts 8 --lr 1e-4 --batch_size 20 --image_size 256 --epochs 15
```

<h5>Testing</h5>

```
python main.py --model_name celeb_8parts --data_path ../datasets/celeba --num_parts 8 --lr 1e-4 --batch_size 20 --image_size 256 --epochs 15
```

<h3>CUB</h3>
<h5>Training</h5>

```
python main.py --model_name CUB_8parts --data_path ../datasets/cub/CUB_200_2011 --num_parts 8 --lr 1e-4 --batch_size 15 --image_size 448 --epochs 28
```

<h5>Testing</h5>

```
python main.py --model_name CUB_8parts --data_path ../datasets/cub/CUB_200_2011 --num_parts 8 --pretrained_model_name CUB_8parts --image_size 448 --warm_start True --only_test True
```

<h3>PartImageNet</h3>
<h5>Training</h5>
```
python main.py --model_name PARTIMAGENET_25parts --data_path ../datasets/partimagenet --num_parts 25 --lr 1e-4 --batch_size 20 --image_size 256 --epochs 20
```

<h5>Testing</h5>

If you find any bugs, please either send me an e-mail or open an issue on GitHub.