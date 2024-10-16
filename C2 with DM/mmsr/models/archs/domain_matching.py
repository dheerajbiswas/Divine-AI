# mmsr/models/archs/domain_matching.py
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models

# Grayscale transformation function
def grayscale_transform(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Whitening and Coloring Transform (WCT) function
def wct(content_features, style_features):
    c_mean = content_features.mean(dim=[2, 3], keepdim=True)
    s_mean = style_features.mean(dim=[2, 3], keepdim=True)

    content_centered = content_features - c_mean
    style_centered = style_features - s_mean

    c_cov = content_centered @ content_centered.permute(0, 1, 3, 2)
    s_cov = style_centered @ style_centered.permute(0, 1, 3, 2)

    c_eigval, c_eigvec = torch.symeig(c_cov, eigenvectors=True)
    s_eigval, s_eigvec = torch.symeig(s_cov, eigenvectors=True)

    whitening = c_eigvec @ torch.diag_embed(torch.sqrt(1 / (c_eigval + 1e-5))) @ c_eigvec.permute(0, 1, 3, 2)
    coloring = s_eigvec @ torch.diag_embed(torch.sqrt(s_eigval + 1e-5)) @ s_eigvec.permute(0, 1, 3, 2)

    transformed_features = coloring @ (whitening @ content_centered) + s_mean
    return transformed_features

# Phase Replacement (PR) function
def phase_replacement(content_feature, stylized_feature):
    content_fft = torch.fft.fft2(content_feature, dim=(-2, -1))
    stylized_fft = torch.fft.fft2(stylized_feature, dim=(-2, -1))

    amplitude_content = torch.abs(content_fft)
    phase_content = torch.angle(content_fft)
    
    amplitude_stylized = torch.abs(stylized_fft)
    
    result = amplitude_stylized * torch.exp(1j * phase_content)
    return torch.fft.ifft2(result, dim=(-2, -1)).real

# Domain Matching Super-Resolution Model
class DomainMatchingSR(nn.Module):
    def __init__(self):
        super(DomainMatchingSR, self).__init__()
        self.encoder = models.vgg19(pretrained=True).features[:21]  # Use VGG19 for feature extraction
    
    def forward(self, lr_image, ref_image):
        # Step 1: Grayscale Transformation
        lr_gray = grayscale_transform(lr_image)
        ref_gray = grayscale_transform(ref_image)

        # Step 2: Feature Extraction (with VGG19)
        lr_features = self.encoder(lr_gray)
        ref_features = self.encoder(ref_gray)

        # Step 3: Whitening and Coloring Transform
        transformed_features = wct(lr_features, ref_features)

        # Step 4: Phase Replacement
        output = phase_replacement(lr_features, transformed_features)

        return output
