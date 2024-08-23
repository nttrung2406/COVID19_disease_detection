import os, sys
sys.path.append('C:/Users/flori/OneDrive/Máy tính/Tai-lieu/HCMUS/Image processing')
from lib import *

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:


        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Create builder config
        config = builder.create_builder_config()

        # Build and return an engine
        engine = builder.build_serialized_network(network, config) 
        return engine

engine = build_engine("C:/Users/flori/OneDrive/Máy tính/Tai-lieu/HCMUS/Image processing/model.onnx")

with open("model.trt", "wb") as f:
    f.write(engine.serialize())