import numpy as np
import math
from model_utils import ModelCommonUtil
import cv2

class Model_GazeEstimation:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None,threshold=0.6):
        #Initialize class
        self.network = None
        self.exec_net = None
        self.model_util = ModelCommonUtil(model_name,device,extensions)
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None

    def load_model(self):
        self.network, self.exec_net =  self.model_util.load_model()
        #Handling inputs and outputs 
        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        self.output_names = [i for i in self.network.outputs.keys()]

    def preprocess_input(self, le_image,re_image):
        input_image = []
        input_image.append(le_image.copy())
        input_image.append(re_image.copy())
        processed_input = self.model_util.preprocess_input(input_image,self.input_shape)
        return processed_input[0] , processed_input[1] #Processed left eye input image , Processed right eye input image

    def predict(self, left_eye, right_eye,hpa):
        processed_le, processed_re = self.preprocess_input(left_eye, right_eye)
        outputs = self.exec_net.infer({'head_pose_angles':hpa, 'left_eye_image':processed_le, 'right_eye_image':processed_re})
        return self.preprocess_output(outputs,hpa)

    def preprocess_output(self, outputs,hpa):
        gaze = outputs[self.output_names[0]].tolist()[0]
        angle = hpa[2]
        cos_angle = math.cos(angle * math.pi / 180.0)
        sin_angle = math.sin(angle * math.pi / 180.0)
        x = gaze[0] * cos_angle + gaze[1] * sin_angle
        y = -gaze[0] *  sin_angle + gaze[1] * cos_angle

        return (x,y), gaze


    def show_annotation(self,gaze,left_eye,right_eye,face,eyes_coords,nose,lips):

        
        h = face.shape[0] * 0.4
        w = face.shape[1] * 0.4

        x, y= int(gaze[0]*w), int(gaze[1]*w)



        le_coords = eyes_coords[0]
        re_coords = eyes_coords[1]
        
        #le_line =cv2.line(left_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
        #cv2.line(le_line, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
        
        #re_line = cv2.line(right_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
        #cv2.line(re_line, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
        
        #face[le_coords[1]:le_coords[3],le_coords[0]:le_coords[2]] = le_line
        #face[re_coords[1]:re_coords[3],re_coords[0]:re_coords[2]] = re_line


        leftEyeMidpoint_start = int(((le_coords[0] + le_coords[2])) / 2)
        leftEyeMidpoint_end = int(((le_coords[1] + le_coords[3])) / 2)
        rightEyeMidpoint_start = int((re_coords[0] + re_coords[2]) / 2)
        rightEyeMidpoint_End = int((re_coords[1] + re_coords[3]) / 2)
        
        

        cv2.arrowedLine(face, 
                        (leftEyeMidpoint_start, leftEyeMidpoint_end), 
                        ((leftEyeMidpoint_start + x), 
                        leftEyeMidpoint_end + (y)),
                        (255, 255, 0), 2)

        cv2.arrowedLine(face, 
                        (rightEyeMidpoint_start, rightEyeMidpoint_End), 
                        ((rightEyeMidpoint_start + x), 
                        rightEyeMidpoint_End + (y)),
                        (255, 255, 0), 2)

   