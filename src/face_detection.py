
import numpy as np
from model_utils import ModelCommonUtil

class Model_FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None,prob_threshold=0.6):
        #Initialize class
        self.prob_threshold=prob_threshold
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
        processed_input = self.preprocess_input(image)
        outputs = self.exec_net.infer({self.input_name:processed_input})
        return self.preprocess_output(outputs,image)

    def preprocess_output(self, outputs,image):
        coords =[]
        outs = outputs[self.output_names][0][0]
        for out in outs:
            conf = out[2]
            if conf>self.prob_threshold:
                x_min=out[3]
                y_min=out[4]
                x_max=out[5]
                y_max=out[6]
                coords.append([x_min,y_min,x_max,y_max])
        return self.crop_face(coords,image)


    def crop_face(self,coords,image):
        
        if (len(coords)==0):
            return 0,0

        coords = coords[0] 
        h = image.shape[0]
        w = image.shape[1]
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32)   
        cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]

        return cropped_face


