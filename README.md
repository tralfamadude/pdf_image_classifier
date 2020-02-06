# Tensorflow Image Classifier

## Overview
This image classifier is based on an example from Google Tensorflow. It includes information on using 
tensorflow serving to do inference in production. The reason for packaging it up in it's own repo 
separate from pdf_trio is because it is useful stand-alone.  

Tensorflow-hub is being used so that pre-trained models curated there can be obtained easily.

## Python Prep
Clone the tf_image_classifier companion repo, and set up the python env.
(We are sssuming conda envs are in use here, adapt as needed.) 
```
git clone https://github.com/tralfamadude/pdf_image_classifier
cd pdf_image_classifier

conda create --name tf_hub python=3.7 
conda activate tf_hub
pip install -r requirements.txt
```

##          Training
### Inputs and Adjustments
By editing this training script, it is possible to try different pre-trained image models, 
alter the learning rate, and number of steps.  Note that different pre-trained models like Inception and Mobilenet can have 
slightly different images sizes.  The images are resized, if necessary, by `tf_hub_image_classifier.py`

Arrange the your training .jpg files to be in 2 directories like:
```
image_training/
image_training/research/
image_training/research/foo.jpg
image_training/research/bar.jpg
image_training/other/
image_training/other/boo.jpg
image_training/other/baz.jpg
```
The directory is specified below to the `train.sh`. 

### Training Command
```
conda activate tf_hub
./train.sh training_data_dir
```
Where the training_data_dir has 2 subdirectories: one for each class, which contain the prepared images. 

When training is done, the name of the directory containing the SavedModel is emitted to stdout. 

## Saved Model Format 
The SavedModel format is needed for tensorflow serving. As an example, here is an `ls -R`  of a saved model:
```
1569277681
./1569277681:
saved_model.pb  variables
./1569277681/variables:
variables.data-00000-of-00001  variables.index
```
We see a top directory `1569277681` which is a Unix epoch that can conveniently stand in for a version number, 
a protocol buffer file `save_model.pb` which contains the serialized graph, and a deeper subdirectory 
named `variables` which contains a large file of parameters `variables.data-00000-of-00001` and an index 
file. 

Saved models are put in a directory so that multiple versions can coexist in the filesystem for tensorflow serving
so that is is possible to make production transitions to new versions without downtime. The highest numeric
version is served by default.  


## Command Line Testing, CPU only
A withheld set of jpg files can be tested on the command line with this script to verify that the training 
went well.
The target directory should contain *.jpg files:
``` 
./test_infer_validation.sh target_dir
```
output is a bit messy, but look for lines starting with your label names, for example:
```
research 0.6590455 validation_data/research//f94399f45fab3ef1e95cf7959a5580aa4b3c1d80.jpg
other 0.3409545 validation_data/research//f94399f45fab3ef1e95cf7959a5580aa4b3c1d80.jpg
```
The model load time and the inference time is printed too. Take heart that with tensorflow serving, the model load 
only happens once, instead of before each inference as here, and inference time for an image is going to be 
faster. This measurement should be considered an upper bound.

