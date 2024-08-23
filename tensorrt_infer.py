import os, sys
sys.path.append('C:/Users/flori/OneDrive/Máy tính/Tai-lieu/HCMUS/Image processing')
from lib import *

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    return h_input, h_output, d_input, d_output, stream

def do_inference(engine, h_input, d_input, h_output, d_output, stream):
    with engine.create_execution_context() as context:
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    return h_output

engine = load_engine("C:/Users/flori/OneDrive/Máy tính/Tai-lieu/HCMUS/Image processing/model.trt")
h_input, h_output, d_input, d_output, stream = allocate_buffers(engine)
output = do_inference(engine, h_input, d_input, h_output, d_output, stream)