Modified DeepLabCut for "AI enabled implantable multichannel wireless telemetry for photodynamic therapy"

Dependencies
Note
conda create -n DLC python=3.7
pip install deeplabcut==2.2b7 
conda install -c conda-forge tensorflow=1.13
pip install wxpython


Usage
1. Download DeepLabCut 2.2b2 (https://github.com/DeepLabCut/DeepLabCut/archive/99155c7f1175c697c04093780fe546a1f17b91a2.zip)
2. Replaced two files in the original projects with the files provided in this repository as follows:
	Deeplabcut/pose_estimation_tensorflow/predict_multianimal.py
	Deeplabcut/pose_estimation_tensorflow/predict_video.py
3. Run main.py

Credits
DeepCutRealTime is an adaptation and extension of DeepLabCut, which is covered by the GNU Lesser General Public License v3.0.
predict_multianimal.py and predict_video.py have been modified based on the original source code files with the same names, respectively.
