# Object TokenGS | Model Card

## **Description:**

The Object TokenGS is a feed-forward neural reconstruction model that takes posed multi-view RGB images as input and predicts a 3D Gaussian Splatting (3DGS) representation for the object.
TokenGS directly regresses 3D Gaussian centers in global coordinates and decouples the number of predicted Gaussians from input image resolution and number of views by using learnable Gaussian tokens in an encoder-decoder Transformer.

### **License/Terms of Use:**
The model is a submodule that follows the terms of [Asset Havester](https://huggingface.co/nvidia/asset-harvester),

### **Deployment Geography:**

Global

### **Use Case:**

Object TokenGS can be used for multi-view 3D object lifting.  It takes multiview images as input, and convert them into 3D Gaussian assets.

### **Release Date:**

This model is on [HuggingFace](https://huggingface.co/nvidia/asset-harvester) and inference script is on [Github](https://github.com/NVIDIA/asset-harvester).

## **References(s):**

- [Asset-Harvester: Turning Autonomous Driving Logs into 3D Assets for Simulation. ]()

## **Model Architecture:**

System architecture details described in white paper above.

## **Input:**

**Input Type(s):** Image
**Input Format(s):** Red, Green, Blue (RGB) images plus camera parameters
**Input Parameters:** Two-Dimensional (2D) images with camera intrinsics and extrinsics; optional timestamp conditioning for dynamic reconstruction
**Other Properties Related to Input:**

- Input includes camera intrinsics and camera extrinsics.
- Images with resolution `512 x 512`

## **Output:**

**Output Type(s):** 3D Gaussian Splatting primitives and rendered RGB images
**Output Format(s):** 3DGS parameter tensors (14 attributes per Gaussian primitive) renderable to novel RGB views via a differentiable Gaussian splatting renderer
**Output Parameters:** 14-dimensional (14D) Gaussian attributes
**Other Properties Related to Output:**

Each Gaussian includes:

- Mean or center: `(x, y, z)`
- Color: `(r, g, b)`
- Scale: `(sx, sy, sz)`
- Opacity: `alpha`
- Rotation: quaternion `(qw, qx, qy, qz)`

Our AI models are designed and optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA hardware and CUDA-enabled software frameworks, the model achieves faster training and inference times compared to CPU-only solutions.

## **Software Integration:**

**Supported Hardware Microarchitecture Compatibility:**

- NVIDIA Ampere
- NVIDIA Blackwell
- NVIDIA Hopper
- NVIDIA Lovelace


**Supported Operating System(s):**

- Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## **Model Version:**

Asset\_Harvester\_GA

## **Training, Testing, and Evaluation Datasets:**

Details described in white paper above.


## **Inference:**

**Acceleration Engine:** PyTorch
**Test Hardware:** NVIDIA A100, H100

## **Ethical Considerations:**

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with the license terms, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards.

Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the generated image or video will not automatically blur or maintain the proportions of image subjects included.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

**Bias**

| Field | Response |
| :---- | :---- |
| Participation considerations from adversely impacted groups [protected classes](https://www.senate.ca.gov/content/protected-classes) in model design and testing: | None |
| Measures taken to mitigate against unwanted bias: | None |

**Explainability**

| Field | Response |
| :---- | :---- |
| Intended Task/Domain: | Multi-view 3D object reconstruction. |
| Model Type: | Transformer |
| Intended Users: | 3D vision, simulation, graphics, and robotics or physical AI researchers and developers. |
| Output | 3D Gaussian Splat representation and rendered novel views. |
| Describe how the model works | Encoder-decoder Transformer with learnable Gaussian tokens directly regresses 3D Gaussian attributes from posed images, trained with rendering and visibility losses. |
| Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of | None |
| Technical Limitations & Mitigation | TokenGS may miss fine-grained geometric details. Quality depends on camera pose quality and multiview coverage, so users should validate outputs and provide sufficient view diversity and accurate camera metadata. |
| Verified to have met prescribed NVIDIA quality standards | Yes |
| Performance Metrics | PSNR, SSIM, LPIPS; additional comparisons under view extrapolation and camera-noise robustness. |
| Potential Known Risks | Reconstruction failures or incomplete geometry may produce misleading renderings or assets. |
| Licensing | The use of the model is governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). |

**Privacy**

| Field | Response |
| :---- | :---- |
| Generatable or reverse engineerable personal data? | No |
| Personal data used to create this model? | No |
| Was consent obtained for any personal data used? | Not Applicable |
| How often is the dataset reviewed? | Before release |
| Is a mechanism in place to honor data subject right of access or deletion of personal data? | Not Applicable |
| If personal data was collected for the development of the model, was it collected directly by NVIDIA? | Not Applicable |
| If personal data was collected for the development of the model by NVIDIA, do you maintain or have access to disclosures made to data subjects? | Not Applicable |
| If personal data was collected for the development of this AI model, was it minimized to only what was required? | Not Applicable |
| Was data from user interactions with the AI model (e.g. user input and prompts) used to train the model? | No |
| Is there provenance for all datasets used in training? | Yes |
| Does data labeling (annotation, metadata) comply with privacy laws? | Yes |
| Is data compliant with data subject requests for data correction or removal, if such a request was made? | Yes |
| Applicable Privacy Policy | [https://www.nvidia.com/en-us/about-nvidia/privacy-policy/](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) |

**Safety & Security**

| Field | Response |
| :---- | :---- |
| Model Application(s): | 3D object reconstruction|
| Describe the life critical impact (if present). | Not Applicable. The model is not intended for direct life-critical decision-making, and outputs should not be used as the sole basis for autonomous vehicle perception, robotics control, or operational safety decisions. Additional validation and testing should be incorporated prior to deployment in real-world production. |
| Use Case Restrictions: | Abide by [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| Model and dataset restrictions: | The Principle of least privilege (PoLP) is applied limiting access for dataset generation and model development. Restrictions enforce dataset access during training |
