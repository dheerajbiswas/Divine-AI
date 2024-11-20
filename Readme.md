# Divine-AI: Implementation of the paper titled 'Bridging the Domain Gap: A Simple Domain Matching Method for Reference-Based Image Super-Resolution in Remote Sensing'

### Team Members
1. Dheeraj Kumar Biswas (P24EE003) 
2. Saransh Chourey (P24EE005) 

### Course Instructor: Dr. Rajesh Kumar Mandotiya

## Description of the work
As per the paper, the domain matching module is integrated with the baseline model of DATSR. As shown in paper, the three algorithms DATSR, C2-Matching and AMSA are used as baseline model for integration of the proposed Domain Matching module. Out of these three models, the DATSR turns out to be best performing algorithm among the others.
That is the reason why its implementation is performed.

## Description of the Domain Matching Architecture

![Screenshot 2024-11-19 192624](https://github.com/user-attachments/assets/cbc91fa4-e6d1-4fb9-9eb0-0aadc489b1b9)

Mainly there are three operations which are taking place in the domain matching process - 
1. Grayscale matching - The grayscale transformation for input photos in order to close the domain gap in the correspondence matching stage. Three channels are averaged to produce a single channel in the grayscale transition. picture before it is entered as input into the encoder, before the matching encoder procedure. Using grayscale conversion successfully removes color information from a photograph, leaving the picture's structural details. This straightforward adjustment reveals that the effect of the distribution gap (e.g., color, brightness) between two pictures is successfully decreased, and the Performance in matching has improved.

2. Whitening and Coloring Transform (WCT) - Whitening and Coloring Transform (WCT) is used in the texture transfer process to lessen the domain gap. WCT compares the style feature map from the pre-trained VGG network with the covariance matrix of the content feature map:

The feature maps of content and style images from a particular VGG network layer are denoted by fc and fs. 
a) Whitening: A feature's covariance matrix can be transformed linearly to become the identity matrix using the Whitening Transform. E is the orthogonal matrix of eigenvectors for rotation, and D is a diagonal matrix for scaling. The sample mean mc is subtracted to get the centered data, or fc. 
b) Coloring: The opposite of whitening is the coloring transform. It creates a certain covariance matrix from the whitened data. Es and Ds are the diagonal matrix of fsfs⊤ and the orthogonal eigenvector matrix. The output is then styled of fs by adding the mean of a style feature ms to fcs.

3. Phase Replacement (PR) - Fourier perspective was used to modify the stylized feature map.
Using style transfer procedures directly could harm the HR Ref image's structural details. We suggest using the Phase Replacement (PR) technique to solve this problem. We maintain the amplitude of the stylized feature map throughout the WCT process while preserving the phase information of the content feature map. This method enables us to preserve the image's structural details and it efficiently reduces the domain gap between input and reference features while maintaining structural information, thus enabling the RefSR model to perform better in terms of SR.



## Description of DATSR model

![framework](https://github.com/user-attachments/assets/84afcf65-c014-4e9b-b7f1-d8adcd1f9046)
The goal of reference-based image super-resolution (RefSR) is to super-resolve low-resolution (LR) images by using auxiliary reference (Ref) images. RefSR has gained a lot of attention lately since it offers a different approach to outperforming single picture SR. However, there are two significant obstacles to solving the RefSR problem: (i) When LR and Ref photos differ greatly, it is hard to match their correspondence; (ii) it is very difficult to figure out how to transfer the required texture from Ref images to make up for the details in LR images.  A deformable attention Transformer, called DATSR, with several scales to address these problems with RefSR. Each scale is made up of a texture feature encoder (TFE), reference-based deformable attention (RDA), and residual feature aggregation (RFA) module. TFE first extracts image transformation (e.g., brightness) insensitive features for LR and Ref images, then RDA can use multiple relevant textures to compensate for more information for LR features, and finally RFA aggregates LR features and relevant textures to produce a more visually appealing result. Extensive trials show that our DATSR outperforms benchmark datasets both statistically and qualitatively.

## Requirements
1. Python 3.8, PyTorch >= 1.7.1
2. CUDA 10.0 or CUDA 10.1
3. GCC 5.4.0

## Dataset
1. DATSR is trained on CUFED5 dataset and tested on RRSSRD dataset
2. Domain matching is trained and tested on RRSSRD dataset

## Execution
Run in command line:
<code>
python dm/test.py 
python dm/train.py
</code>

## Results - 


## References 
1. Min, Jeongho, et al. "Bridging the Domain Gap: A Simple Domain Matching Method for Reference-based Image Super-Resolution in Remote Sensing." IEEE Geoscience and Remote Sensing Letters (2023).
2. Cao, Jiezhang, et al. "Reference-based image super-resolution with deformable attention transformer." European conference on computer vision. Cham: Springer Nature Switzerland, 2022.
