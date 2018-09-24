import onnx
import onnx_caffe2.backend as backend

# Prepare the inputs, here we use numpy to generate some random inputs for demo purpose
import numpy as np
img = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Load the ONNX model
model = onnx.load('TFeat_tf.onnx')
# Run the ONNX model with Caffe2
outputs = backend.prepare(model, [img])

