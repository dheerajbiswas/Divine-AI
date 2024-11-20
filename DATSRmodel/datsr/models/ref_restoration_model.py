import importlib
import logging
import os.path as osp
from collections import OrderedDict

import mmcv
import torch

import models.networks as networks
import utils.metrics as metrics
from utils import ProgressBar, tensor2img, img2tensor

from .sr_model import SRModel
import pdb

loss_module = importlib.import_module('models.losses')
logger = logging.getLogger('base')
psnr_list = []


class RefRestorationModel(SRModel):
    def __init__(self, opt):
        super(RefRestorationModel, self).__init__(opt)

        # Grayscale transformation
        self.grayscale = torch.mean(image, dim=0, keepdim=True)

        # Define network for feature extraction
        self.net_extractor = networks.define_net_extractor(opt)
        self.net_extractor = self.model_to_device(self.net_extractor)
        self.print_network(self.net_extractor)

        # Phase replacement components (whitening and color transformation)
        self.phase_replacement = self.define_phase_replacement(opt)

        # Initialize other networks and losses (as in the original code)
        ...

    def define_phase_replacement(self, opt):
        """
        Define phase replacement network or operations.
        This will include whitening and color transformations.
        """
        c_mean = content_features.mean(dim=[2, 3], keepdim=True)
        s_mean = style_features.mean(dim=[2, 3], keepdim=True)
        content_centered = content_features - c_mean
        style_centered = style_features - s_mean

        c_cov = content_centered @ content_centered.transpose(-2, -1)
        s_cov = style_centered @ style_centered.transpose(-2, -1)

        c_eigval, c_eigvec = torch.symeig(c_cov, eigenvectors=True)
        s_eigval, s_eigvec = torch.symeig(s_cov, eigenvectors=True)

        whitening = c_eigvec @ torch.diag_embed(torch.sqrt(1 / (c_eigval + 1e-5))) @ c_eigvec.transpose(-2, -1)
        coloring = s_eigvec @ torch.diag_embed(torch.sqrt(s_eigval + 1e-5)) @ s_eigvec.transpose(-2, -1)

        # return   coloring @ (whitening @ content_centered) + s_mean

        # Example: Whitening and Color Transform Layers
        return torch.nn.Sequential(
            torch.nn.BatchNorm2d(opt['feature_channels'], coloring),
            torch.nn.Conv2d(opt['feature_channels'], opt['feature_channels'], kernel_size=1),
        )

    def preprocess_images(self, img_lr, img_ref):
        """
        Apply grayscale and whitening transformation to images.
        """
        img_lr_gray = self.grayscale(img_lr)
        img_ref_gray = self.grayscale(img_ref)

        # Normalize grayscale images
        img_lr_gray = F.normalize(img_lr_gray, mean=[0.5], std=[0.5])
        img_ref_gray = F.normalize(img_ref_gray, mean=[0.5], std=[0.5])

        return img_lr_gray, img_ref_gray

    def feed_data(self, data):
        """
        Override feed_data to include grayscale transformation and preprocessing.
        """
        self.img_in_lq = data['img_in_lq'].to(self.device)
        self.img_ref = data['img_ref'].to(self.device)

        # Apply grayscale transformation
        self.img_in_lq, self.img_ref = self.preprocess_images(self.img_in_lq, self.img_ref)

        self.gt = data['img_in'].to(self.device)  # Ground truth
        self.match_img_in = data['img_in_up'].to(self.device)

    def optimize_parameters(self, step):
        """
        Override optimize_parameters to include phase replacement and transformations.
        """
        # Extract features from inputs
        features = self.net_extractor(self.match_img_in, self.img_ref)

        # Apply phase replacement transformations
        transformed_features = self.phase_replacement(features)

        # Pass transformed features to generator
        self.output = self.net_g(self.img_in_lq, transformed_features, self.img_ref)

        # Compute loss and optimize
        ...

    def test(self):
        """
        Test method with grayscale and transformations.
        """
        self.net_g.eval()
        with torch.no_grad():
            self.features = self.net_extractor(self.match_img_in, self.img_ref)
            transformed_features = self.phase_replacement(self.features)
            self.output = self.net_g(self.img_in_lq, transformed_features, self.img_ref)

        self.net_g.train()