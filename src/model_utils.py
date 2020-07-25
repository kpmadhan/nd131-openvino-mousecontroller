import cv2
import sys
import logging as log
import numpy as np
from openvino.inference_engine import IENetwork,IECore


class ModelCommonUtil:

    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_name = model_name
        self.device = device
        self.extensions= extensions

    def load_model(self):
      
        # Getting the reference of the model
        model_structure = self.model_name
        model_weights = self.model_name.split('.')[0]+'.bin'

        self.ie = IECore()
        if (self.extensions):
            self.ie.add_extension(extension_path=self.extensions, device_name=self.device)
        self.network = self.ie.read_network(model=model_structure, weights=model_weights)

        self.check_model()

        return self.network , self.ie.load_network(network=self.network, device_name=self.device,num_requests=1)


    def check_model(self):
        #Check for unsupported layers 
        if self.device == "CPU":     
            supported_layers = self.ie.query_network(network=self.network, device_name=self.device)  
            unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(unsupported_layers) != 0:
                log.error("[ERROR] Unsupported layers found: {}".format(unsupported_layers))
                sys.exit(1)


    def preprocess_input(self, images,input_shape):
        # Resize and change channels 
        processed_inputs = []
        for image in images: 
            p_frame = cv2.resize(image,(input_shape[3], input_shape[2]))
            p_frame = p_frame.transpose(2, 0, 1)
            p_frame = p_frame.reshape(1, *p_frame.shape)
            processed_inputs.append(p_frame)
        return processed_inputs


