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

import os
import os.path
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.util import img_as_ubyte
from tqdm import tqdm

from deeplabcut.pose_estimation_tensorflow.nnet import predict_multianimal as predict
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal, auxfun_videos

from itertools import product
from scipy.spatial import distance
from skimage.draw import circle, line_aa
from deeplabcut.pose_estimation_tensorflow.lib.inferenceutils import convertdetectiondict2listoflist

import re 
import math
import matplotlib.pyplot as plt
import networkx
import scipy.spatial.distance
from matplotlib.animation import FFMpegWriter
from networkx.algorithms.matching import max_weight_matching

def AnalyzeMultiAnimalVideo(
    video,
    DLCscorer,
    trainFraction,
    cfg,
    dlc_cfg,
    sess,
    inputs,
    outputs,
    pdindex,
    save_as_csv,
    destfolder=None,
    c_engine=False,
    robust_nframes=False,
    realTime=False,
    showEstimationResult=True,
    topView=False,
    saveVideo=False,
):
    if realTime:
        print("Starting to analyze stream")
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise IOError(
                "Camera could not be opened. Please check that a camera is properly connected."
            )

        if robust_nframes:
            nframes = auxfun_videos.get_nframes_robust(video)
            duration = auxfun_videos.get_duration(video)
            fps = nframes / duration
        else:
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = nframes / fps
        size = (int(cap.get(4)), int(cap.get(3)))

        ny, nx = size
        print(
            "Duration of video [s]: ",
            round(duration, 2),
            ", recorded with ",
            round(fps, 2),
            "fps!",
        )
        print(
            "Overall # of frames: ",
            nframes,
            " found with (before cropping) frame dimensions: ",
            nx,
            ny,
        )
        if saveVideo:
            videooutname = video.replace(".mp4", "labeled.mp4")
            # prev_backend = plt.get_backend()
            # plt.switch_backend("agg")
            # dpi = 100
            # fig = plt.figure(frameon=False, figsize=(nx / dpi, ny / dpi))
            # ax = fig.add_subplot(111)
            # writer = FFMpegWriter(fps=fps, codec="h264")  
            # writer.saving(fig, videooutname, dpi=dpi)  
            frameProcessed = 0
            frameLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out = cv2.VideoWriter(videooutname, cv2.CAP_FFMPEG , fourcc = cv2.VideoWriter_fourcc(*'DIVX'), fps=fps, frameSize = (nx, ny))

        start = time.time()

        all_jointnames = cfg["multianimalbodyparts"]
        numjoints = len(all_jointnames)
        bpts = range(numjoints)

        colorclass = plt.cm.ScalarMappable(cmap=cfg["colormap"])
        C = colorclass.to_rgba(np.linspace(0, 1, numjoints))
        colors = (C[:, :3] * 255).astype(np.uint8)

        countForDebug = 0
        dotsize = cfg["dotsize"]
        pcutoff = cfg["pcutoff"]
        print("Starting to extract posture")
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                directions = np.zeros(4) # Up, down, left, right

                if cfg["cropping"]:
                    frame = img_as_ubyte(
                        frame[cfg["y1"] : cfg["y2"], cfg["x1"] : cfg["x2"]]
                    )
                else:
                    frame = img_as_ubyte(frame)

                PredicteData = predict.get_detectionswithcosts(
                    frame,
                    dlc_cfg,
                    sess,
                    inputs,
                    outputs,
                    outall=False,
                    nms_radius=dlc_cfg.nmsradius,
                    det_min_score=dlc_cfg.minconfidence,
                    c_engine=c_engine,
                )

                bpts = range(numjoints)
                dets = convertdetectiondict2listoflist(PredicteData, bpts)
                # for i, det in enumerate(dets):
                #     color = colors[i]
                #     for x, y, p, _ in det:
                #         if p > pcutoff:
                #             rr, cc = circle(y, x, dotsize, shape=(ny, nx))
                #             frame[rr, cc] = color

                indexSnout = all_jointnames.index("snout")
                indexTail = all_jointnames.index("tail")
                bodyparts = list()
                for i, det in enumerate(dets):
                    if i == indexSnout or i == indexTail:
                        bodypart = list()
                        for x, y, p, _ in det:
                            if p > pcutoff:
                            # if p > 0.2:
                                bodypart.append([x, y])
                        bodyparts.append(bodypart)

                # inverted = False
                # if len(bodyparts[0]) >= len(bodyparts[1]):
                #     a = [*range(0, len(bodyparts[0]))]
                #     b = [*range(0, len(bodyparts[1]))]
                # else:
                #     a = [*range(0, len(bodyparts[1]))]
                #     b = [*range(0, len(bodyparts[0]))]
                #     inverted = True
                # bijectiveMappings = list(list(zip(p, r)) for (r, p) in zip(itertools.repeat(b), itertools.permutations(a)))

                maxMappingDistance = (nx / 4)
                minMappingDistance = 40
                backgroundValueThreshold = 60
                backgroundCountThreshold = 30
                if (len(bodyparts[0]) > 0) and (len(bodyparts[1]) > 0):
                    distanceMatrix = distance.cdist(bodyparts[0], bodyparts[1])
                    distanceMatrix = 1/(distanceMatrix + 1)
                    # normalizedMatrix = distanceMatrix/maxMappingDistance
                    # distanceMatrix = 1 - normalizedMatrix

                    G = networkx.Graph()
                    for i in range(len(bodyparts[0])):
                        for j in range(len(bodyparts[1])):
                            rr, cc, val = line_aa(int(bodyparts[0][i][1]), int(bodyparts[0][i][0]), int(bodyparts[1][j][1]), int(bodyparts[1][j][0]))
                            # mid = int(len(rr)/2)
                            # colorWeight = ((255-np.median(frame[rr, cc][:, 0])) + (255-np.median(frame[rr, cc][:, 1])) + (255-np.median(frame[rr, cc][:, 2])))/3
                            # colorWeight = np.median(frame[rr, cc][:, 0]) + np.median(frame[rr, cc][:, 1]) + np.median(frame[rr, cc][:, 2])
                            numberOfBackgroundPixels = 0
                            for c in range(len(frame[rr,cc][0])):
                                numberOfBackgroundPixels += sum(frame[rr, cc][:, c] > backgroundValueThreshold)
                            if (numberOfBackgroundPixels < backgroundCountThreshold) and (distanceMatrix[i, j] > (1/maxMappingDistance)) and (distanceMatrix[i, j] < (1/minMappingDistance)):
                                G.add_edge("a"+str(i), "b"+str(j), weight = distanceMatrix[i, j])
                            # G.add_edge("a"+str(i), "b"+str(j), weight=np.linalg.norm(np.array(bodyparts[0][i])-np.array(bodyparts[1][j])))
                            # font = cv2.FONT_HERSHEY_SIMPLEX
                            # cv2.putText(frame,'mean:' + str(np.mean(val)) + ', var: ' + str(np.var(val)), (cc[mid],rr[mid]), font, 1,(255,255,255), 2)
                            # frame[rr, cc] = colors[1]
                                    
                    mappings = max_weight_matching(G) 
                    for s in mappings:
                        indexSnout = int(re.findall(r'\d+', np.sort(s)[0])[0])
                        rr, cc = circle(bodyparts[0][indexSnout][1], bodyparts[0][indexSnout][0], dotsize, shape=(ny, nx))
                        if  showEstimationResult:
                            frame[rr, cc] = colors[0]

                        indexTail = int(re.findall(r'\d+', np.sort(s)[1])[0])
                        rr, cc = circle(bodyparts[1][indexTail][1], bodyparts[1][indexTail][0], dotsize, shape=(ny, nx))
                        if  showEstimationResult:
                            frame[rr, cc] = colors[-1]

                        rr, cc, val = line_aa(int(bodyparts[0][indexSnout][1]), int(bodyparts[0][indexSnout][0]), int(bodyparts[1][indexTail][1]), int(bodyparts[1][indexTail][0]))
                        if showEstimationResult:
                            frame[rr, cc] = colors[1]

                        xSnout = bodyparts[0][indexSnout][0]
                        ySnout = (ny-bodyparts[0][indexSnout][1])
                        xTail = bodyparts[1][indexTail][0]
                        yTail = (ny-bodyparts[1][indexTail][1])
                        
                        angleCurrent = math.degrees(math.atan2((ySnout-yTail), (xSnout-xTail)))
                        mid = int(len(rr)/2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        if showEstimationResult:
                            cv2.putText(frame, str(round(angleCurrent, 2)), (cc[mid],rr[mid]), font, 0.6 ,(255,255,255), 2)
                        if (angleCurrent <= 45) and (angleCurrent > -45):
                            directions[3] += 1
                        elif (angleCurrent <= 135) and (angleCurrent > 45):
                            directions[0] += 1
                        elif (angleCurrent <= -135) or (angleCurrent > 135):
                            directions[2] += 1
                        else:
                            directions[1] += 1
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                if showEstimationResult:
                    cv2.putText(frame, 'Up: ' + str(directions[0]) + '   Down: ' + str(directions[1]) + '   Left: ' + str(directions[2]) + '   Right: ' + str(directions[3]), (int(nx/3)+20, 20), font, 0.6 ,(255,255,255), 2)
                
                # for i, det in enumerate(bodyparts):
                #     color = colors[i]
                #     for x, y, p in det:
                #         rr, cc = circle(y, x, dotsize, shape=(ny, nx))
                #         frame[rr, cc] = color

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
                if saveVideo:
                    out.write(frame)
                    frameProcessed += 1
                    print("frame processed: " + str(frameProcessed) + " total frames: " + str(frameLength))
                # else:
                #     # countForDebug+=1
                #     # if countForDebug > (57+25+32+22+20):
                #     #     if cv2.waitKey(0) == ord('q'):
                #     #         break
                #     cv2.imshow('frame', frame)
                #     if cv2.waitKey(0) == ord('q'):
                #         break
            else:
                break

        PredicteData["metadata"] = {
            "nms radius": dlc_cfg.nmsradius,
            "minimal confidence": dlc_cfg.minconfidence,
            "PAFgraph": dlc_cfg.partaffinityfield_graph,
            "all_joints": [[i] for i in range(len(dlc_cfg.all_joints))],
            "all_joints_names": [
                dlc_cfg.all_joints_names[i] for i in range(len(dlc_cfg.all_joints))
            ],
            "nframes": nframes,
        }
        stop = time.time()

        if cfg["cropping"] == True:
            coords = [cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"]]
        else:
            coords = [0, nx, 0, ny]

        dictionary = {
            "start": start,
            "stop": stop,
            "run_duration": stop - start,
            "Scorer": DLCscorer,
            "DLC-model-config file": dlc_cfg,
            "fps": fps,
            "batch_size": dlc_cfg["batch_size"],
            "frame_dimensions": (ny, nx),
            "nframes": nframes,
            "iteration (active-learning)": cfg["iteration"],
            "training set fraction": trainFraction,
            "cropping": cfg["cropping"],
            "cropping_parameters": coords,
        }

        if saveVideo:
            print("endOfOperation")
            out.release()
        else: 
            cv2.destroyWindow('frame')
    else:
        """ Helper function for analyzing a video with multiple individuals """

        print("Starting to analyze % ", video)
        vname = Path(video).stem
        videofolder = str(Path(video).parents[0])
        if destfolder is None:
            destfolder = videofolder
        auxiliaryfunctions.attempttomakefolder(destfolder)
        dataname = os.path.join(destfolder, vname + DLCscorer + ".h5")

        if os.path.isfile(dataname.split(".h5")[0] + "_full.pickle"):
            print("Video already analyzed!", dataname)
        else:
            print("Loading ", video)
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                raise IOError(
                    "Video could not be opened. Please check that the path is valid."
                )

            if robust_nframes:
                nframes = auxfun_videos.get_nframes_robust(video)
                duration = auxfun_videos.get_duration(video)
                fps = nframes / duration
            else:
                nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                duration = nframes / fps
            size = (int(cap.get(4)), int(cap.get(3)))

            ny, nx = size
            print(
                "Duration of video [s]: ",
                round(duration, 2),
                ", recorded with ",
                round(fps, 2),
                "fps!",
            )
            print(
                "Overall # of frames: ",
                nframes,
                " found with (before cropping) frame dimensions: ",
                nx,
                ny,
            )
            start = time.time()

            print("Starting to extract posture")
            if int(dlc_cfg["batch_size"]) > 1:
                PredicteData, nframes = GetPoseandCostsF(
                    cfg,
                    dlc_cfg,
                    sess,
                    inputs,
                    outputs,
                    cap,
                    nframes,
                    int(dlc_cfg["batch_size"]),
                    c_engine=c_engine,
                )
            else:
                PredicteData, nframes = GetPoseandCostsS(
                    cfg, dlc_cfg, sess, inputs, outputs, cap, nframes, c_engine=c_engine
                )

            stop = time.time()

            if cfg["cropping"] == True:
                coords = [cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"]]
            else:
                coords = [0, nx, 0, ny]

            dictionary = {
                "start": start,
                "stop": stop,
                "run_duration": stop - start,
                "Scorer": DLCscorer,
                "DLC-model-config file": dlc_cfg,
                "fps": fps,
                "batch_size": dlc_cfg["batch_size"],
                "frame_dimensions": (ny, nx),
                "nframes": nframes,
                "iteration (active-learning)": cfg["iteration"],
                "training set fraction": trainFraction,
                "cropping": cfg["cropping"],
                "cropping_parameters": coords,
            }
            metadata = {"data": dictionary}
            print("Saving results in %s..." % (destfolder))

            auxfun_multianimal.SaveFullMultiAnimalData(PredicteData, metadata, dataname)


def GetPoseandCostsF(
    cfg, dlc_cfg, sess, inputs, outputs, cap, nframes, batchsize, c_engine
):
    """ Batchwise prediction of pose """
    strwidth = int(np.ceil(np.log10(nframes)))  # width for strings
    batch_ind = 0  # keeps track of which image within a batch should be written to
    batch_num = 0  # keeps track of which batch you are at
    ny, nx = int(cap.get(4)), int(cap.get(3))
    if cfg["cropping"]:
        print(
            "Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file."
            % (cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"])
        )
        nx = cfg["x2"] - cfg["x1"]
        ny = cfg["y2"] - cfg["y1"]
        if nx > 0 and ny > 0:
            pass
        else:
            raise Exception("Please check the order of cropping parameter!")
        if (
            cfg["x1"] >= 0
            and cfg["x2"] < int(cap.get(3) + 1)
            and cfg["y1"] >= 0
            and cfg["y2"] < int(cap.get(4) + 1)
        ):
            pass  # good cropping box
        else:
            raise Exception("Please check the boundary of cropping!")

    frames = np.empty(
        (batchsize, ny, nx, 3), dtype="ubyte"
    )  # this keeps all frames in a batch
    pbar = tqdm(total=nframes)
    counter = 0
    step = max(10, int(nframes / 100))

    PredicteData = {}
    # initializing constants
    dist_grid = predict.make_nms_grid(dlc_cfg.nmsradius)
    stride, halfstride = dlc_cfg.stride, dlc_cfg.stride * 0.5
    num_joints = dlc_cfg.num_joints
    det_min_score = dlc_cfg.minconfidence

    num_idchannel = dlc_cfg.get("num_idchannel", 0)
    while cap.isOpened():
        if counter % step == 0:
            pbar.update(step)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cfg["cropping"]:
                frames[batch_ind] = img_as_ubyte(
                    frame[cfg["y1"] : cfg["y2"], cfg["x1"] : cfg["x2"]]
                )
            else:
                frames[batch_ind] = img_as_ubyte(frame)

            if batch_ind == batchsize - 1:
                # PredicteData['frame'+str(counter)]=predict.get_detectionswithcosts(frame, dlc_cfg, sess, inputs, outputs, outall=False,nms_radius=dlc_cfg.nmsradius,det_min_score=dlc_cfg.minconfidence)
                D = predict.get_batchdetectionswithcosts(
                    frames,
                    dlc_cfg,
                    dist_grid,
                    batchsize,
                    num_joints,
                    num_idchannel,
                    stride,
                    halfstride,
                    det_min_score,
                    sess,
                    inputs,
                    outputs,
                )
                for l in range(batchsize):
                    # pose = predict.getposeNP(frames,dlc_cfg, sess, inputs, outputs)
                    # PredicteData[batch_num*batchsize:(batch_num+1)*batchsize, :] = pose
                    PredicteData[
                        "frame" + str(batch_num * batchsize + l).zfill(strwidth)
                    ] = D[l]

                batch_ind = 0
                batch_num += 1
            else:
                batch_ind += 1
        else:
            nframes = counter
            print("Detected frames: ", nframes)
            if batch_ind > 0:
                # pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs) #process the whole batch (some frames might be from previous batch!)
                # PredicteData[batch_num*batchsize:batch_num*batchsize+batch_ind, :] = pose[:batch_ind,:]
                D = predict.get_batchdetectionswithcosts(
                    frames,
                    dlc_cfg,
                    dist_grid,
                    batchsize,
                    num_joints,
                    num_idchannel,
                    stride,
                    halfstride,
                    det_min_score,
                    sess,
                    inputs,
                    outputs,
                    c_engine=c_engine,
                )
                for l in range(batch_ind):
                    # pose = predict.getposeNP(frames,dlc_cfg, sess, inputs, outputs)
                    # PredicteData[batch_num*batchsize:(batch_num+1)*batchsize, :] = pose
                    PredicteData[
                        "frame" + str(batch_num * batchsize + l).zfill(strwidth)
                    ] = D[l]
            break
        counter += 1

    pbar.close()
    PredicteData["metadata"] = {
        "nms radius": dlc_cfg.nmsradius,
        "minimal confidence": dlc_cfg.minconfidence,
        "PAFgraph": dlc_cfg.partaffinityfield_graph,
        "all_joints": [[i] for i in range(len(dlc_cfg.all_joints))],
        "all_joints_names": [
            dlc_cfg.all_joints_names[i] for i in range(len(dlc_cfg.all_joints))
        ],
        "nframes": nframes,
        "c_engine": c_engine,
    }
    return PredicteData, nframes


def GetPoseandCostsS(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes, c_engine):
    """ Non batch wise pose estimation for video cap."""
    strwidth = int(np.ceil(np.log10(nframes)))  # width for strings
    if cfg["cropping"]:
        print(
            "Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file."
            % (cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"])
        )
        nx = cfg["x2"] - cfg["x1"]
        ny = cfg["y2"] - cfg["y1"]
        if nx > 0 and ny > 0:
            pass
        else:
            raise Exception("Please check the order of cropping parameter!")
        if (
            cfg["x1"] >= 0
            and cfg["x2"] < int(cap.get(3) + 1)
            and cfg["y1"] >= 0
            and cfg["y2"] < int(cap.get(4) + 1)
        ):
            pass  # good cropping box
        else:
            raise Exception("Please check the boundary of cropping!")

    PredicteData = {}  # np.zeros((nframes, 3 * len(dlc_cfg['all_joints_names'])))
    pbar = tqdm(total=nframes)
    counter = 0
    step = max(10, int(nframes / 100))
    while cap.isOpened():
        if counter % step == 0:
            pbar.update(step)

        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cfg["cropping"]:
                frame = img_as_ubyte(
                    frame[cfg["y1"] : cfg["y2"], cfg["x1"] : cfg["x2"]]
                )
            else:
                frame = img_as_ubyte(frame)
            PredicteData[
                "frame" + str(counter).zfill(strwidth)
            ] = predict.get_detectionswithcosts(
                frame,
                dlc_cfg,
                sess,
                inputs,
                outputs,
                outall=False,
                nms_radius=dlc_cfg.nmsradius,
                det_min_score=dlc_cfg.minconfidence,
                c_engine=c_engine,
            )
        else:
            nframes = counter
            break
        counter += 1

    pbar.close()
    PredicteData["metadata"] = {
        "nms radius": dlc_cfg.nmsradius,
        "minimal confidence": dlc_cfg.minconfidence,
        "PAFgraph": dlc_cfg.partaffinityfield_graph,
        "all_joints": [[i] for i in range(len(dlc_cfg.all_joints))],
        "all_joints_names": [
            dlc_cfg.all_joints_names[i] for i in range(len(dlc_cfg.all_joints))
        ],
        "nframes": nframes,
    }

    # print(PredicteData)
    return PredicteData, nframes
