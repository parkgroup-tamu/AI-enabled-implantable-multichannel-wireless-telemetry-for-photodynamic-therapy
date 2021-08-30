"""
Adapted from DeepLabCut2.0 Toolbox (deeplabcut.org)
by 
Hyun-Myung Woo, larcwind@tamu.edu
Byung-Jun Yoon, bjyoon@ece.tamu.edu

DeepLabCut2.0 Toolbox (deeplabcut.org)
by
A Mathis, alexander.mathis@bethgelab.org | https://github.com/AlexEMG/DeepLabCut
T Nath, nath@rowland.harvard.edu | https://github.com/meet10may
M Mathis, mackenzie@post.harvard.edu | https://github.com/MMathisLab

Licensed under GNU Lesser General Public License v3.0
"""

"""
How to use:

deeplabcut.analyze_videos(myPath, input, realTime = True, showEstimationResult =  True, topView = True, saveVideo = False)

1. myPath (string): Path to the config.yaml.
2. input (int or string): Input 0 allows the program to run in a realtime manner with the main webcam.
                          For debugging purpose, a vidio file can be used as input with full path in string format.
3. realTime (True/False): If realTime is True, the modified module is executed instead of the module from the original DLC.

# For debugging
4. showEstimationResult (True/False): If showEstimationResult and saveVideo are True, the estimation results are marked in the video file.
5. topView (True/False): If topView is True, the program assumes that the webcam is installed on the top of the cage.
6. saveVideo (True/False): If saveVideo is True, the estimation results is saved as a video file.
"""

import deeplabcut

myPath = './PE5-HM-2020-10-20/config.yaml'
input = './sample.mp4'
realTime = True
showEstimationResult = True
topView = True
saveVideo = True

deeplabcut.analyze_videos(myPath, input, realTime = realTime, showEstimationResult =  showEstimationResult, topView = topView, saveVideo = saveVideo)