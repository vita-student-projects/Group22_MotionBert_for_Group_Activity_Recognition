# Group22_MotionBert_for_Group_Activity_Recognition


## Volleyball dataset:
The dataset we are using is the volleyball dataset that can be downloaded here: https://drive.google.com/drive/folders/1rmsrG1mgkwxOKhsr-QYoi9Ss92wQmCOS?usp=sharing

Once downloaded, put your dataset in the videos folder

## Data Preprocessing:
(If the data used for testing is already in the pkl. format, you can skip this step.)

In order to extract the poses, we use a 2D pose extraction tool using HRNet from pyskl.

MODIFY THIS
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
You can now start the pose extraction.


## Activity Recognition