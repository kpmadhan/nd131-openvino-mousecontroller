
import numpy as np
from model_utils import ModelCommonUtil
import cv2

class Model_FacialLandmark:
    '''
    Class for the Face Landmark Detection Model.
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
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape

    
    def preprocess_input(self, image):
        input_image = []
        input_image.append(image.copy())
        processed_input = self.model_util.preprocess_input(input_image,self.input_shape)
        return processed_input[0]

    def predict(self, image):
        processed = self.preprocess_input(image)
        outputs = self.exec_net.infer({self.input_name:processed})
        return self.preprocess_output(outputs,image)

    def preprocess_output(self, outputs,image):
        outs = outputs[self.output_names][0]
        leye_x = outs[0].tolist()[0][0]
        leye_y = outs[1].tolist()[0][0]
        reye_x = outs[2].tolist()[0][0]
        reye_y = outs[3].tolist()[0][0]
        nose_x = outs[4].tolist()[0][0]
        nose_y = outs[5].tolist()[0][0]
        llip_x = outs[6].tolist()[0][0]
        llip_y = outs[7].tolist()[0][0]
        rlip_x = outs[8].tolist()[0][0]
        rlip_y = outs[9].tolist()[0][0]
        return self.preprocess_coords((leye_x, leye_y, reye_x, reye_y),(nose_x,nose_y),(llip_x,llip_y,rlip_x,rlip_y),image) 
        
    def preprocess_coords(self,eyecoords,nose,lips,image):
        
        h = image.shape[0]
        w = image.shape[1]

        eye_area=10

        eyecoords = eyecoords * np.array([w, h, w, h])
        eyecoords = eyecoords.astype(np.int32)
        le_xmin=eyecoords[0]-eye_area
        le_ymin=eyecoords[1]-eye_area
        le_xmax=eyecoords[0]+eye_area
        le_ymax=eyecoords[1]+eye_area
        
        re_xmin=eyecoords[2]-eye_area
        re_ymin=eyecoords[3]-eye_area
        re_xmax=eyecoords[2]+eye_area
        re_ymax=eyecoords[3]+eye_area
        
        left_eye =  image[le_ymin:le_ymax, le_xmin:le_xmax]
        right_eye = image[re_ymin:re_ymax, re_xmin:re_xmax]
        eye_coords = [[le_xmin,le_ymin,le_xmax,le_ymax], [re_xmin,re_ymin,re_xmax,re_ymax]]
        
        return left_eye, right_eye, eye_coords , (nose * np.array([w, h])).astype(np.int32) , (lips * np.array([w, h, w, h])).astype(np.int32)
    
    def show_annotation(self,face,eyes_coords,nose,lips):
        cv2.circle(face, self.midpoint(eyes_coords[0]) , 12, (0,255,0), 2) # left eye
        cv2.circle(face, self.midpoint(eyes_coords[1]), 12, (0,255,0), 2)  #right eye
        cv2.circle(face, tuple(nose), 6, (0,255,0), 2)  #nose
   
    def midpoint(self,coords):
        return (int(round((coords[0] + coords[2])/2)),int(round((coords[1] + coords[3])/2)))   