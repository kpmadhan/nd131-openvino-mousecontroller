import cv2
import time
import os
import numpy as np
import common_utils as utils
import logging as log

from face_detection import Model_FaceDetection as fd_mdl
from facial_landmarks_detection import Model_FacialLandmark as fld_mdl
from gaze_estimation import Model_GazeEstimation as ge_mdl
from head_pose_estimation import Model_HeadPoseEstimation as hpe_mdl

from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder


log.basicConfig(
    level=log.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler("debug.log"),
        log.StreamHandler()
    ]
)

#@profile
def feedInput(input):
	
    feeder = None
    input_type = "cam"
    if input != 'CAM':
        assert os.path.isfile(input)
        if input.endswith(('.jpg', '.bmp', '.png')):
            input_type = "image"
        else:
            input_type = "video"
        feeder = InputFeeder(input_type=input_type, input_file=input)
    else:
        feeder = InputFeeder(input_type=input_type)

    return feeder 


def init_models(args):
    loader = []
    models = {
        0: {'model_path': args.face_detection_model,'method' : fd_mdl}, 
        1: {'model_path': args.facial_landmarks_detection_model,'method' : fld_mdl}, 
        2: {'model_path': args.head_pose_estimation_model,'method' : hpe_mdl}, 
        3: {'model_path': args.gaze_estimation_model, 'method' : ge_mdl}}

    isModelExist(models) # if not , exit(1)
    for i in range(4):
        invoke = models[i]['method']
        loader.append(invoke(models[i]['model_path'], args.device, args.cpu_extension, args.threshold))
    return  tuple(loader) #face_detec, fac_land, head_pose, gaze_est

def main():
    
    args = build_argparser().parse_args()

    #Init video feeder
    feeder = feedInput(args.input)
    feeder.load_data()

    mouse_controller = MouseController('medium','fast')

 
    face_detec, fac_land, head_pose, gaze_est = init_models(args)

    mdl_start = cv2.getTickCount() 

    face_detec.load_model()
    fac_land.load_model()
    head_pose.load_model()
    gaze_est.load_model()

    load_time = utils.timeLapse(mdl_start)
    
    count = 0
    predict_time =[]
    start=time.time()
    for r, frame in feeder.next_batch():
       
        if not r:
        	break
        count+=1

        key = cv2.waitKey(60)
        if key==27:
            break  

        pdt_start = cv2.getTickCount() 
        face = face_detec.predict(frame)
        hp_pred = head_pose.predict(face)
        left_eye, right_eye, eyes_coords ,  nose , lips = fac_land.predict(face)
        mouse_coords, gaze = gaze_est.predict(left_eye, right_eye,hp_pred)
        predict_time.append(utils.timeLapse(pdt_start))

        

        if (len(args.verbose)>0):
            new_frame = face.copy()
            if 'landmark' in args.verbose or 'all' in args.verbose:
                fac_land.show_annotation(new_frame,eyes_coords,nose,lips)
            if 'headpose' in args.verbose or 'all' in args.verbose:
                head_pose.show_annotation(new_frame,hp_pred)
            if 'gaze' in args.verbose or 'all' in args.verbose:
                gaze_est.show_annotation(gaze,left_eye,right_eye,new_frame,eyes_coords,nose,lips)

        elif(len(args.verbose) == 0):
            new_frame = frame.copy()
            

        cv2.imshow('display',cv2.resize(new_frame,(500,500)))
        if(count%5==0):
            mouse_controller.move(mouse_coords[0],mouse_coords[1])
	

    
    log.info('######################################################')
    log.info('# Average Inference Time (all model)                                            ::  {:.3f} ms'.format(np.mean(predict_time)))
    log.info('# Total Model Load Time  (all model)                                            ::  {:.3f} ms'.format(load_time))
    log.info('######################################################')

    inference_time=time.time()-start
    fps=count/inference_time
    

    if args.path is not None:
        with open("{}.txt".format(args.path), "w") as f:
            f.write(str(load_time)+'\n')
            f.write(str(inference_time)+'\n')
            f.write(str(fps)+'\n')


    cv2.destroyAllWindows()
    feeder.close()

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    
    parser.add_argument("-f", "--face_detection_model", required=True, type=str,
                        help="Path to .xml file of the face detection model")

    parser.add_argument("-l", "--facial_landmarks_detection_model", required=True, type=str,
                        help="Path to .xml file of the facial landmarks detection model")

    parser.add_argument("-hp", "--head_pose_estimation_model", required=True, type=str,
                        help="Path to .xml file of the head pose estimation model")

    parser.add_argument("-g", "--gaze_estimation_model", required=True, type=str,
                        help="Path to .xml file of the gaze estimation model")

    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or enter 'cam' to work with webcam")

    parser.add_argument("-t", "--threshold", required=False, type=float, default=0.6,
                        help="Probability threshold for the face detection model.")

    parser.add_argument("-v", "--verbose", required=False, type=str,nargs='+', default=[],
                        help="See detailed output of selected models. "
                        "Valid inputs: 'landmark', 'headpose' , 'gaze' or 'all' ")
    
    parser.add_argument("-d", "--device", required=False,type=str, default="CPU",
                        help="Target device to infer on: "
                             "Valid inputs: CPU, GPU, FPGA or MYRIAD.")

    parser.add_argument("-ext", "--cpu_extension", required=False,type=str, default=None,
                        help="Path to the CPU extension")
    
    parser.add_argument('--path', default=None)

    return parser

def isModelExist(model_info):
    for _, model in model_info.items():
        for key in model:
            if( key is 'model_path'):
                if not os.path.isfile(model[key]):
                    print("[ERROR] file does not exist")
                    exit(1)


if __name__ == '__main__':
    main()


