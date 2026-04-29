# Mask2Former Overview | Model Card

## **Description:**

The AV Object Mask2Former is a model that performs object instance segmentation tasks. It was trained on object-centric AV images.

This model is used in the Asset Harvester System.

### **License/Terms of Use:**

GOVERNING TERMS: The use of the model is governed by the [NVIDIA Software and Model Evaluation License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

### **Deployment Geography:**

Global

### **Use Case:**

The model can be used for segmenting object-centric AV images.  Given an image cropped from AV video, it output binary mask of the object in the center of the image.

### **Release Date:**

HuggingFace 03/16/26

## **Reference:**

[Bowen Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng,+B), [Ishan Misra](https://arxiv.org/search/cs?searchtype=author&query=Misra,+I), [Alexander G. Schwing](https://arxiv.org/search/cs?searchtype=author&query=Schwing,+A+G), [Alexander Kirillov](https://arxiv.org/search/cs?searchtype=author&query=Kirillov,+A), [Rohit Girdhar](https://arxiv.org/search/cs?searchtype=author&query=Girdhar,+R), Masked-attention Mask Transformer for Universal Image Segmentation, [https://arxiv.org/abs/2112.01527](https://arxiv.org/abs/2112.01527).

## **Model Architecture:**

* Fully Convolutional Networks (FCNs) + Transformer

## **Input:**

* **Input Type(s):** Image
* **Input Format(s):** Red, Green, Blue (RGB)
* **Input Parameters:** The input parameters to this model are 2D query features (X0) and 3D image features (Kl, Vl) with dimensions N x C, where N is the number of query features and C is the number of channels.
* **Other Properties Related to Input:** Spatial resolution of image features: 32, 16, 8.

## **Output:**

* **Output Type(s):** Image
* **Output Format(s):** Binary mask
* **Output Parameters:** The output parameters of this model are the predicted mask for each query, with dimensions of the input query features being N x C, where N is the number of query features and C is the number of channels.
* **Other Properties Related to Output:** Resolution: H1=H=32, H2=H=16, H3=H=8 and W1=W=32, W2=W=16

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## **Software Integration:**

**Runtime Engine(s):**
PyTorch

**Supported Hardware Microarchitecture Compatibility:**

* NVIDIA Ampere
* NVIDIA Blackwell
* NVIDIA Hopper
* NVIDIA Lovelace

**[Preferred/Supported] Operating System(s):**
Linux

## **Model Version(s):**

V1

## **Training, Testing, and Evaluation Datasets:**

The AV Object Mask2former was trained, tested, and evaluated using NVIDIA proprietary AV dataset.

| Dataset names | Size and content | Training partition | Test partition |
| :---- | :---- | :---- | :---- |
| Internal Nvidia AV dataset | Posed images of 278k objects | 83% (cross validation) | 17% |

### Internal NVIDIA AV dataset

**Link:** N/A

**Data Collection Method:** Sensors

**Labeling Method by Dataset:** Automated. The labels we collected are binary masks of objects in the images.

**Properties**: This dataset was collected using sensors mounted on the NVIDIA fleet and was auto-labeled using a third party tool to ensure high-quality annotations.

## **Inference:**

**Engine:**
PyTorch

**Test Hardware:**
A6000

## **Ethical Considerations:**

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

**Bias**

| Field | Response |
| :---- | :---- |
| Participation considerations from adversely impacted groups [protected classes](https://www.senate.ca.gov/content/protected-classes) in model design and testing: | None |
| Measures taken to mitigate against unwanted bias: | None |

**Explainability**

| Field | Response |
| :---- | :---- |
| Intended Domain | Advanced Driver Assistance Systems |
| Model Type: | Object detection and Instance segmentation |
| Intended Users: | Autonomous Vehicles developers enhancing and improving Neural Reconstruction pipelines. |
| Output | Image Segmentation |
| Describe how the model works | The model takes as an input an image, and outputs a segmentation mask of the image |
| Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of | None |
| Technical Limitations | The system does not guarantee a 100% success rate. The model was trained mostly on vehicles and would not perform well on pedestrians, cyclists, or other non-vehicular objects and struggles with small objects |
| Verified to have met prescribed NVIDIA quality standards | Yes |
| Performance Metrics | Intersection over Union (IOU) |
| Potential Known Risks | AV and robotics developers should be aware that this model cannot guarantee a 100% success rate. In cases of unsuccessful generation, the output may not possess an accurate real-world representation of the asset and should not be relied upon in safety-critical simulations. |
| Licensing | The use of the model is governed by the [NVIDIA Software and Model Evaluation License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). |

**Privacy**

| Field | Response |
| :---- | :---- |
| Generatable or reverse engineerable personal data? | No |
| Personal data used to create this model? | No |
| How often is the dataset reviewed? | Before release |
| Is there provenance for all datasets used in training? | Yes |
| Does data labeling (annotation, metadata) comply with privacy laws? | Yes |
| Is data compliant with data subject requests for data correction or removal, if such a request was made? | Yes |
| Applicable Privacy Policy | [https://www.nvidia.com/en-us/about-nvidia/privacy-policy/](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) |

**Safety & Security**

| Field | Response |
| :---- | :---- |
| Model Application(s): | Object detection and Segmentation |
| Describe the life critical impact (if present). | N/A \- The model should not be deployed in a vehicle to perform life-critical tasks. |
| Use Case Restrictions: | The use of the model is governed by the [NVIDIA Software and Model Evaluation License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). |
| Model and dataset restrictions: | The Principle of least privilege (PoLP) is applied limiting access for dataset generation and model development. Restrictions enforce dataset access during training, and dataset license constraints adhered to. |
