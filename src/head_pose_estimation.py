'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from model_utils import ModelCommonUtil

class Model_HeadPoseEstimation:
    '''
    Class for the Head Pose Estimation Model.
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
        self.output_names = [i for i in self.network.outputs.keys()]

    def preprocess_input(self, image):
        input_image = []
        input_image.append(image)
        processed_input = self.model_util.preprocess_input(input_image,self.input_shape)
        return processed_input[0]

    def predict(self, image):
        processed = self.preprocess_input(image)
        outputs = self.exec_net.infer({self.input_name:processed})
        return self.preprocess_output(outputs)

    def preprocess_output(self, outputs):
        out = []
        out.append(outputs['angle_y_fc'].tolist()[0][0])
        out.append(outputs['angle_p_fc'].tolist()[0][0])
        out.append(outputs['angle_r_fc'].tolist()[0][0])
        return out

    def show_annotation(self, new_frame,hp_pred):
        cv2.putText(new_frame, "yaw :{:.2f} | pitch :{:.2f} | roll :{:.2f}".format(hp_pred[0],hp_pred[1],hp_pred[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 255), 1)
	        