# Multiview Diffusion (Sana-based) | Model Card

## **Description:**

The multiview diffusion model was trained on AV object images with a SANA base model. The model is conditioned on image input and outputs images of the same object in different viewpoints. It doesn't support text input.

This model is used as part of the Asset Harvester GA.

### **License/Terms of Use:**

### Governing Terms: Use of this model system is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) .

### **Deployment Geography:**

Global

### **Use Case:**

The multiview diffusion model takes a set of posed images as input and outputs 16 images from different viewpoints of the same input vehicle. The goal of it is to provide the 16 output images as input for three-dimensional (3D) reconstruction to generate 3D assets.

### **Release Date:**

HuggingFace

## **Reference(s):**

**Asset-Harvester: Turning Autonomous Driving Logs into 3D Assets for Simulation.** *NVIDIA white paper.*
\[later we replace it with our paper link\]

## **Model Architecture:**

**Architecture Type:** Linear Diffusion Transformer

**Network Architecture:** Sparse View Linear-attention Diffusion Transformer, as described in our white paper,
with a Deep Compression Autoencoder (DC-AE) for efficient high-resolution image generation. C-RADIO for image conditioning signal.

## **Input:**

**Input Type(s):** Up to 4 Images (Adjustable via config parameter)

**Input Format(s):** Red, Green, Blue (RGB)

**Input Parameters:** Two-Dimensional (2D)

**Other Properties Related to Input:** Camera matrices of images

## **Output:**

**Output Type(s):** 16 Images

**Output Format(s):** Red, Green, Blue (RGB)

**Output Parameters:** Two-Dimensional (2D)

**Other Properties Related to Output:** Camera poses of images

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## **Software Integration:**

**Runtime Engine(s):**
PyTorch

**Supported Hardware Microarchitecture Compatibility:**
NVIDIA Ampere

**[Preferred/Supported] Operating System(s):**
Linux

## **Model Version(s):**

v1

## **Training, Testing, and Evaluation Datasets:**

The model was trained, tested, and finetuned using an Objaverse subset internal AV data, and Omniverse 3D assets (synthetic images).

| Dataset names | Size and content | Training partition | Test partition |
| :---- | :---- | :---- | :---- |
| Nvidia Proprietary AV dataset | Posed images of 278k objects | 83% (cross validation) | 17% |
| Omniverse 3D assets | 200 3D assets of objects | 100% | 0% |
| Objaverse | 80k assets collected under commercially viable Creative Commons licenses, | 100% | 0% |

### Objaverse Commercially Viable Subset under CC licenses

**Link:** https://objaverse.allenai.org
**Data Collection Method:** Synthetic 3D assets aggregated from various open-source and licensed sources
**Labeling Method by Dataset:** Hybrid: Human and Automated
**Properties:** This dataset consists of a diverse set of over 80,000 synthetic 3D object models spanning everyday items, animals, tools, and complex structures. Each model is rendered into multi-view 2D images with associated camera poses, materials, and mesh properties.

### Nvidia Proprietary AV dataset

**Data Collection Method:** Sensors

**Labeling Method by Dataset:** Human

**Properties**: This dataset was collected using sensors mounted on the NVIDIA fleet and was manually labeled by a team of human annotators to ensure high-quality annotations.

### Omniverse 3D assets

**Data Collection Method:** Human

**Labeling Method by Dataset:** Human

**Properties**: This dataset was collected using humans that create 3D assets.

## **Inference:**

**Engine:** PyTorch>=2.0.0

**Test Hardware:**
We tested on H100, A100, A6000 and RTX4090. Inference time using 1XA100 is 7 seconds per 16 images.

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
| Model Type: | Multiview creation |
| Intended Users: | Autonomous Vehicles developers enhancing and improving Neural Reconstruction pipelines. |
| Output | 16 images |
| Describe how the model works | The model takes as an input an image (up to 4\) and outputs 16 multiviews of the vehicles detected in the original image |
| Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of | None |
| Technical Limitations | The system does not guarantee a 100% success rate. It cannot fully guarantee the safety and controllability of the generated image content. Additionally, challenges remain in certain complex cases, such as text rendering and the generation of faces and hands. |
| Verified to have met prescribed NVIDIA quality standards | Yes |
| Performance Metrics | Peak signal-to-noise ratio (PSNR), FID (Frechet Inception Distance), CLIPScore |
| Potential Known Risks | AV and robotics developers should be aware that this model cannot guarantee a 100% success rate. In cases of unsuccessful generation, the output may not possess an accurate real-world representation of the asset and should not be relied upon in safety-critical simulations. |
| Licensing | The use of the model is governed by the [NVIDIA Software and Model Evaluation License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). |

**Privacy**

| Field | Response |
| :---- | :---- |
| Generatable or reverse engineerable personal data? | No |
| Personal data used to create this model? | Yes |
| Was consent obtained for any personal data used? | Yes |
| Is a mechanism in place to honor data subject right of access or deletion of personal data? | Yes |
| If personal data was collected for the development of the model, was it collected directly by NVIDIA? | No |
| If personal data was collected for the development of the model by NVIDIA, do you maintain or have access to disclosures made to data subjects? | N/A |
| If personal data was collected for the development of this AI model, was it minimized to only what was required? | Yes |
| Was data from user interactions with the AI model (e.g. user input and prompts) used to train the model? | Yes |
| How often is the dataset reviewed? | Before release |
| Is there provenance for all datasets used in training? | Yes |
| Does data labeling (annotation, metadata) comply with privacy laws? | Yes |
| Is data compliant with data subject requests for data correction or removal, if such a request was made? | Yes |
| Applicable Privacy Policy | [https://www.nvidia.com/en-us/about-nvidia/privacy-policy/](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) |

**Safety & Security**

| Field | Response |
| :---- | :---- |
| Model Application(s): | Multiview creation |
| Describe the life critical impact (if present). | N/A \- The model should not be deployed in a vehicle to perform life-critical tasks. |
| Use Case Restrictions: | The use of the model is governed by the [NVIDIA Software and Model Evaluation License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). |
| Model and dataset restrictions: | The Principle of least privilege (PoLP) is applied limiting access for dataset generation and model development. Restrictions enforce dataset access during training, and dataset license constraints adhered to. |
