# UMich_STATS_507_Final_Proj Proposal

## Overview

Pneumonia remains a leading cause of morbidity and mortality worldwide, especially among young children and elderly patients. Chest X-ray imaging (CXR) is one of the most widely used and cost-effective tools for pneumonia screening, but manual interpretation is time-consuming and subject to inter-reader variability. Recent advances in deep learning have shown promising results in automating pneumonia detection from CXR images using convolutional neural networks (CNNs) and, more recently, Vision Transformers (ViTs)[1].

In this project, I propose to build a robust pneumonia detection system based on transfer learning from Hugging Face vision models[2], and to systematically study the impact of data augmentation and model architecture on both accuracy and robustness.

I will use a publicly available chest X-ray dataset **mmenendezg/pneumonia_x_ray**[3] from the Hugging Face Hub, which is a processed version of the well-known UCSD pediatric pneumonia dataset. This dataset contains approximately 5,856 frontal chest X-ray images split into train, validation, and test sets (4187 / 1045 / 624 images), with each image labeled as either normal or pneumonia.

The main goals of the project are:

- Build strong baselines for pneumonia vs. normal classification using a ResNet-18 CNN and a Vision Transformer model.
- Design and evaluate custom data augmentation pipelines (varying strength and type of augmentation) and analyze their impact on performance, overfitting, and robustness to distribution shift.
- Compare CNN and ViT architectures in terms of accuracy, robustness to image corruptions (noise, blur, low contrast, occlusion), and qualitative interpretability (Grad-CAM visualization).

By the end of this project, I expect to obtain quantitative insights into:

- How different augmentation strategies affect model performance in a small/medium-sized medical dataset.
- Whether ViT models provide any clear advantage over ResNet-style CNNs in this setting.
- How robust these models are to realistic image perturbations and whether data augmentation improves robustness.

## Prior Work

### Related work

Deep learning has become the dominant paradigm for automated pneumonia detection from chest X-ray (CXR) images. Early work based on convolutional neural networks (CNNs) such as ResNet and DenseNet demonstrated that CNNs trained with transfer learning from ImageNet can achieve radiologist-level or near–radiologist-level performance on large public datasets including ChestX-ray14 and pediatric pneumonia collections, and can therefore serve as effective decision-support tools in clinical screening workflows[4].ubsequent studies have refined these architectures and training strategies, but have also highlighted persistent challenges such as class imbalance, overfitting on relatively small institutional datasets, and substantial performance degradation under distribution shift when models are evaluated on data from different scanners, hospitals, or patient populations[5].

More recently, Vision Transformers (ViTs) have been introduced to image recognition and rapidly adopted in medical imaging applications. Dosovitskiy et al. showed that pure transformer architectures, when pre-trained at scale, can match or surpass state-of-the-art CNNs on natural-image benchmarks.Building on this idea, several works have explored ViT variants for CXR classification and broader thoracic disease detection, generally reporting performance comparable to or slightly better than CNN baselines and suggesting that the global self-attention mechanism may capture long-range dependencies that are difficult for localized convolutional filters[6].However, comparative analyses between ViTs and CNNs for pneumonia detection specifically remain limited, and most studies emphasize aggregate metrics such as accuracy or AUC rather than robustness, data-efficiency, or calibration.

A second line of work focuses on data augmentation and robustness for deep models in medical imaging. Because labeled medical images are costly to obtain, augmentation is widely used as a form of implicit regularization. Surveys synthesize a broad spectrum of geometric and photometric transformations—random flips, rotations, crops, noise injection, and contrast changes—and show that appropriate augmentation can substantially improve generalization across modalities such as mammography, ultrasound, and CXR[7]. More recent studies extend this idea by designing augmentation schemes that explicitly target robustness to domain shift or adversarial perturbations, for example through corruption benchmarks or adversarially sampled training distributions[8]. Nevertheless, the interaction between augmentation strength, dataset size, and model architecture (CNN versus ViT) in the specific context of pneumonia detection from CXRs has not been systematically characterized.

Finally, interpretability methods have been developed to make CXR classifiers more transparent. Gradient-based localization techniques such as Grad-CAM produce class-specific heatmaps that highlight image regions most influential for a model’s prediction. These visual explanations have been used to verify whether networks concentrate on clinically relevant pulmonary structures or instead exploit spurious cues such as laterality markers and image borders, and can help identify failure modes and dataset bias[9]. In this project, we build on this line of work by using Grad-CAM to qualitatively analyze the attention patterns of CNN and ViT models trained under different augmentation and robustness settings for pneumonia detection.

### Potential methods for this project

Based on these literature, suitable methods include:

- Transfer learning with ResNet-18 and ViT models(From Hugging Face)
- Custom data augmentation pipelines using
- Robustness evaluation under synthetic corruptions (Gaussian noise, motion blur, low contrast, occlusion)
- Interpretability using Grad-CAM to visualize model attention on chest X-rays

## Preliminary Results and Planned Approach

At the time of this proposal, I have not yet run full experiments, but I have examined the dataset and planned the modeling pipeline:

The chosen dataset[3] consists of 5,856 chest X-ray images from independent patients, split into train (4187), validation (1045), and test (624) sets, with two labels: normal and pneumonia. The dataset originates from a widely used public CXR collection and is known to have some class imbalance, with typically more pneumonia images than normal ones. Potential challenges include limited dataset size compared to natural image benchmarks, possible label noise and variability in image quality and Risk of overfitting, especially for large models like ViT.

### Preliminary data analysis

I conducted an initial exploratory data analysis of the training split using Python. The training set contains 4,187 images, with 1,080 normal and 3,107 pneumonia cases, confirming a strong class imbalance. This motivates the use of data augmentation and evaluation metrics such as F1-score and recall for the pneumonia class, rather than relying solely on accuracy.
<img width="500" height="400" alt="Figure_1" src="https://github.com/user-attachments/assets/639d6b00-2cd7-48e3-beaf-c23c3f1bc83b" />

To better understand the visual characteristics of each class, I inspected random examples from the training set. Normal images typically show clear lung fields and regular thoracic anatomy, while pneumonia images often exhibit diffuse opacities, consolidation, and other abnormalities. These qualitative differences support the feasibility of a supervised learning approach.

<img width="1250" height="300" alt="Figure_2" src="https://github.com/user-attachments/assets/3d06d8aa-4236-4c47-ba40-a571325ce523" />

<img width="1250" height="300" alt="Figure_3" src="https://github.com/user-attachments/assets/a00eab8c-4240-49f6-a883-8f12cafbeec1" />

I also analyzed the image sizes. All training images are stored at a fixed resolution of 500 × 500 pixels, as indicated by the degenerate histograms of widths and heights. This simplifies preprocessing, since they can be uniformly resized or center-cropped to the 224 × 224 input resolution required by the ResNet and ViT models.

<img width="1000" height="400" alt="Figure_4" src="https://github.com/user-attachments/assets/61fe0494-5b78-400d-ae54-1c1ab6a4181c" />

Finally, I examined the pixel-intensity distribution for a random subset of training images. Intensities span the full 0–255 range with a large mass at low values and a broad peak in the mid-to-high range, which is consistent with typical chest X-ray contrast. These observations support the use of standard normalization and moderate photometric data augmentation (e.g., slight brightness and contrast jitter) in the later training pipeline.

<img width="500" height="400" alt="Figure_5" src="https://github.com/user-attachments/assets/0675a039-8369-4765-9e03-552d1b43cd40" />

### Basic model and tools

To implement the baseline and subsequent models, I will use Python data-processing tools such as pandas and NumPy, and visualization libraries like matplotlib and seaborn, while computing evaluation metrics including accuracy, precision, recall, F1-score, and ROC–AUC with scikit-learn. The core baseline will be a ResNet-18 classifier initialized with ImageNet pre-trained weights, with its final fully connected layer replaced by a two-class head for normal vs. pneumonia prediction and trained initially with simple preprocessing (resize, center crop, normalization) and no heavy augmentation. Building on this baseline, I will incorporate new tools from the Hugging Face and PyTorch ecosystems, including datasets to load the CXR dataset, transformers to fine-tune vision models such as microsoft/resnet-18 and google/vit-base-patch16-224, and PyTorch/Torchvision together with Grad-CAM utilities to implement custom data-augmentation pipelines and interpretability analyses.

## Project Deliverables

- If things go as expected:
  - training and evaluating ResNet-18 and ViT models on the chest X-ray dataset  
  - implementing custom data-augmentation pipelines with different strengths  
  - running basic robustness experiments under several image corruptions  
  - producing Grad-CAM visualizations for selected test images
  - A main set of quantitative results, including:
    - a head-to-head comparison of ResNet-18 (with and without augmentation) vs. ViT in terms of accuracy, F1, and AUC  
    - at least one ablation study on training-set size (e.g. 25%, 50%, 100%) with/without augmentation

- If there is extra time after the core pieces are finished(well, I think if I can finish the thing above is a great success:) ):
  - run a more systematic study of augmentation strength (light vs. medium vs. strong) and its effect on overfitting  
  - expand the robustness experiments to more corruption types and plot full performance-vs-severity curves  
  - add richer interpretability analysis (more Grad-CAM examples or additional methods) comparing CNN and ViT failure cases.

## Timeline

- Week 1–2
  - Load the pneumonia CXR dataset from Hugging Face and perform EDA (class distribution, sample visualization, already done as shown above).
  - Implement data preprocessing and the baseline ResNet-18 model without heavy augmentation.
  - Train and evaluate the baseline model; log initial metrics and learning curves.

- Week 3-4
  - Design and implement multiple data augmentation pipelines (light / medium / strong).
  - Train ResNet-18 with different augmentation settings and perform data size ablations.
  - Integrate a ViT model from Hugging Face and fine-tune it on the same dataset and augmentation pipeline.
  - Implement synthetic corruption functions (noise, blur, contrast, occlusion) and run robustness experiments for selected models.
  - Begin implementing Grad-CAM visualizations.

- Week 5
  - Continue optimize hyperparameters(if time alowed).
  - Write the project report.
  - Clean up the GitHub repository, ensure code reproducibility, and prepare the final submission.







# References List

[1] Md. R. Hasan, S. M. Azmat Ullah, and S. Md. Rabiul Islam, “Recent advancement of deep learning techniques for pneumonia prediction from chest X-ray image,” Medical Reports, vol. 7, p. 100106, Oct. 2024. doi:10.1016/j.hmedic.2024.100106 

[2] “Google/VIT-base-patch16-224 · hugging face,” google/vit-base-patch16-224 · Hugging Face, https://huggingface.co/google/vit-base-patch16-224 (accessed Nov. 11, 2025). 

[3] M. Menendez, “Mmenendezg/pneumonia_x_ray · datasets at hugging face,” mmenendezg/pneumonia_x_ray · Datasets at Hugging Face, https://huggingface.co/datasets/mmenendezg/pneumonia_x_ray (accessed Nov. 11, 2025). 

[4] M.-J. Tsai and Y.-H. Tao, “Machine learning based common radiologist-level pneumonia detection on chest x-rays,” 2019 13th International Conference on Signal Processing and Communication Systems (ICSPCS), pp. 1–7, Dec. 2019. doi:10.1109/icspcs47537.2019.9008684 

[5] P. Rajpurkar et al., “CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning,” arXiv.org, https://arxiv.org/abs/1711.05225 (accessed Nov. 11, 2025). 

[6] A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv.org, https://doi.org/10.48550/arXiv.2010.11929 (accessed Nov. 11, 2025). 

[7] C. Shorten and T. M. Khoshgoftaar, “A survey on image data augmentation for Deep Learning,” Journal of Big Data, vol. 6, no. 1, Jul. 2019. doi:10.1186/s40537-019-0197-0 

[8] S. Yang et al., “Image Data Augmentation for Deep Learning: A Survey,” arXiv.org, https://arxiv.org/abs/2204.08610 (accessed Nov. 12, 2025). 

[9] R. R. Selvaraju et al., “Grad-cam: Visual explanations from deep networks via gradient-based localization,” 2017 IEEE International Conference on Computer Vision (ICCV), pp. 618–626, Oct. 2017. doi:10.1109/iccv.2017.74



