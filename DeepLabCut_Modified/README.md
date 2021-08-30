# Modified DeepLabCut for<br/>"AI enabled implantable multichannel wireless telemetry for photodynamic therapy"

1. Dependencies<br/>
The modifications have been made based on the DeepLabCut (DLC) version 2.2b7.<br/>
The versions of the important software & packages are as follows:<br/>
Python: 3.7<br/>
deeplabcut: 2.2b7<br/>
tensorflow: 1.13<br/>

2. Usage
	1. Download DeepLabCut 2.2b2 (https://github.com/DeepLabCut/DeepLabCut/archive/99155c7f1175c697c04093780fe546a1f17b91a2.zip)<br/>
	2. Replace two files in the original projects with the files provided in this repository as follows:<br/>
	Deeplabcut/pose_estimation_tensorflow/predict_multianimal.py<br/>
	Deeplabcut/pose_estimation_tensorflow/predict_video.py<br/>
	3. Run main.py

3. Credits
DeepLabCut is under the GNU Lesser General Public License v3.0.
predict_multianimal.py and predict_video.py have been modified based on the original source code files with the same names, respectively.
