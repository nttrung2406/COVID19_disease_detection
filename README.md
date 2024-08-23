# COVID19 disease classification

This is the project for COVID 19 classification challenge. Datasets can be found on Kaggle:

https://www.kaggle.com/datasets/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset

The aims of the project are:

+ Implement an CNN model with 5 layers to predict COVID19, PNEUMONIA or NORMAL.

+ Convert pytorch model to ONNX model and TENSORRT to boost the inference speed using NVIDIA gpu.

+ Comparing among Pytorch, ONNX and TENSORRT model.

**Prerequisite:**

+ Must have NVIDA GPU

**Installation:**

Install TENSORRT can follow this direction:

https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip

**Visulize model with https://netron.app/ :**

![model onnx](https://github.com/user-attachments/assets/3e9372bc-3880-4ec5-9f4a-50183f4aeb5d)


**Performance:**

![output](https://github.com/user-attachments/assets/3ec4538d-433e-4f3e-b91f-48e16c248432)
