# Group22_MotionBert_for_Group_Activity_Recognition
## Introduction
The goal of this project is to identify action based on the movements and poses of the body joints captured from an RGB video.
The project involves developing a deep learning model that can take 2D or 3D poses as input would then be trained to recognize different actions based on the sequence of skeletal joint positions.
The main goal would be to detect the action of pedestrians and then in a future step to be able to predict their actions. For an autonomous vehicule, it will allow to ensure the safety of everyone/better interaction between pedestrians and vehicules.

In this project, we will mainly focus on a subproblem of activity recognition: group activity recognition. Group activity recognition refers to the collective behavior of a group of people, resulted from the individual actions of the persons and their interactions.

This project will be divided in 2 parts: Pose extraction from Volleyball dataset and then finetuning on MotionBert model


In order to train the action recognition model of MotionBert, we used a well-known dataset for group activity recognition: Volleyball dataset (https://github.com/mostafa-saad/deep-activity-rec).

The next steps would be to train our model with pedestrian images. 



# Extraction of poses
## Volleyball dataset:
The dataset we are using is the volleyball dataset that can be downloaded here: https://drive.google.com/drive/folders/1rmsrG1mgkwxOKhsr-QYoi9Ss92wQmCOS?usp=sharing

Once downloaded, put your dataset in the videos folder

## Data Preprocessing:
(If the data used for testing is already in the pkl. format, you can skip this step.)

In order to extract the poses, we use a 2D pose extraction tool using HRNet from pyskl.

Installation of pyskl
```shell
git clone https://github.com/vita-student-projects/Group22_MotionBert_for_Group_Activity_Recognition.git
cd pose_extraction/pyskl

conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
```
### The first step of the data preprocessing will be to generate a list file containing the path of each videos that will be used.
```shell
cd videos
python generate_video_list.py
```
Then run the pose extractions with pyskl thanks to the code below:
(Note that this code works for data that is in jpg format. If the video used are in mp4, modify the file pyskl/tools/data/custom_2d_skeleton.py at line 218 to call extract_frame(anno['filename']) instead of extract_frame_jpg(anno['filename']).)
```shell
cd ../pose_extraction/pyskl
python3 tools/data/custom_2d_skeleton.py --video-list <path_to_video_list> --out <output_path> --non-dist
```
It will generate a pkl file.
We need one more step before starting the activity recognition: we need to modify our pkl file, to make it usable in motionbert.
To do this, go in the /pose_extraction directory , 
and modify the file merge_pkl.py to specify the input and output path
Then run:
```shell
python3 merge_pkl.py
```
It will merge the different pkl files in the input folder (if there are mutiple), generate new entries in the pkl file, like the label. It will perform tracking to always have the same people affected to the same entry, remove clips where the number of players is insufficient (here <8), and finally it removes the non-players by looking at annotations file in the videos folder (the annotations comes with the volleyball dataset).
You can now start the pose extraction.


## Activity Recognition

To solve this problem, we decided to use an existing model that has the capability to detect action based on the body joints.
We used one of the most recent models that has shown one of the best accuracy (97%) in terms action recognition : MotionBert (https://github.com/Walter0807/MotionBERT/tree/main). This model uses a transformer to estimate the 3d poses of the person, to do a mesh recovery and to detect action.
The model is first train on the H3.6M dataset and then finetune on the NTU60 dataset

The challenge of this project would be to implement in the code several people and then be able to extract a correlation between the skeleton of each person. Indeed, MotionBert has shown these results for NTU dataset that usually only include 2 people max. 


### Installation 
```shell
conda create -n motionbert python=3.7 anaconda
conda activate motionbert
# Please install PyTorch according to your CUDA version.
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```
Before finetuning, place the following files (present in the main) at their corresponding locations:
First download the best_epochs.bin and latest_epochs.bin from this drive : https://drive.google.com/drive/u/1/folders/1o8i3nzRWv58D1Hgl5Aruik8FyX8_CTSu
best_epochs.bin : checkpoint/action/MB_train_volley_xsub
latest_epochs.bin : checkpoint/pretrain/MB_release

Moreover, change the data_path (line 127) in the train_action_volley.py according to your location of your pkl files.
Pay attention to the number of frames in your video, you may need to modify it in the config file : MB_ft_volley_xsub.yaml

### Finetuning part for action recognition
To run the script, enters this line in your .sh file. A GPU is necessary for the finetuning part.
```shell
python train_action_volley.py \
--config configs/action/MB_ft_volley_xsub.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/action/FT_MB_release_MB_ft_volley_xsub
```

### Evaluation 
```shell
python train_action_volley.py \
--config configs/action/MB_train_volley_xsub.yaml \
--evaluate checkpoint/action/MB_train_volley_xsub/best_epoch.bin 
```


# Results
You can find below the results that we obtained after the finetuning
| Trials        | Number epochs | Learning rate (head)| Number hidden layers| Accuracy-1 (%)|
| ------------- |:-------------:| -----:|-----:|------:|
| 1     | 300 | 0.005 | 2 | 43.9|
| 2     | 116      |  0.01 | 5 |39|

The first trial consists of of using the same parameters used for NTU60 by MotionBert. The accuracy increases very few so we decided to increase the learning rate but it we didn't see any changes. The accuracy stagnates around 44% after 200 epochs.
For the second trial, we can see that we obtain encouraging results by increasing the number of hidden layers. Unfortunately, it took a lot of time to run and we were not able to reach 300 epochs.


