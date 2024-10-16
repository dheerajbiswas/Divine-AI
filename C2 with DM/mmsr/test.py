import os
import torch
from torch import nn
from collections import OrderedDict
from mmsr.models.archs import domain_matching  # Import Domain Matching module
from mmsr.utils import imwrite, tensor2img, get_root_logger, mkdir_and_rename
from mmsr.models import build_model
from mmsr.data import create_dataset, create_dataloader
from mmsr.options import dict2str, parse
import cv2
import numpy as np


def test_pipeline(opt):
    '''Main testing pipeline to run C2-matching with Domain Matching preprocessing'''

    # Setup environment and model
    torch.backends.cudnn.benchmark = True
    logger = get_root_logger()
    logger.info('Set log level to {}'.format(logger.level))
    logger.info(dict2str(opt))

    # Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in {}: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    # Build model
    model = build_model(opt)

    # Initialize Domain Matching model
    domain_matching_model = domain_matching.DomainMatchingSR().to(opt['device'])

    for test_loader in test_loaders:
        dataset = test_loader.dataset

        # Begin testing
        for idx, val_data in enumerate(test_loader):
            img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
            logger.info(f'Testing {img_name}')

            # Send low-resolution (LQ) and reference (Ref) images to device
            model.feed_data(val_data, need_GT=True)

            # Apply Domain Matching preprocessing before SR
            lr_img = val_data['LQ'].to(opt['device'])
            ref_img = val_data['ref'].to(opt['device'])

            # Use Domain Matching model to preprocess LQ and Ref
            lr_transformed = domain_matching_model(lr_img, ref_img)

            # Set the preprocessed LQ image back to the model and perform inference
            model.feed_data({'LQ': lr_transformed, 'ref': ref_img}, need_GT=True)
            model.test()

            visuals = model.get_current_visuals(need_GT=True)
            sr_img = tensor2img([visuals['rlt']])  # Convert output tensor to image
            gt_img = tensor2img([visuals['GT']]) if 'GT' in visuals else None

            # Save SR results
            save_img_path = os.path.join(opt['path']['results_root'], img_name, '{:s}.png'.format(img_name))
            imwrite(sr_img, save_img_path)

            if gt_img is not None:
                # If ground truth is available, evaluate metrics like PSNR, SSIM
                psnr, ssim = calculate_psnr_ssim(sr_img, gt_img, crop_border=opt['crop_border'], input_order='HWC')
                logger.info(f'{img_name} - PSNR: {psnr:.6f} dB, SSIM: {ssim:.6f}')
            else:
                logger.info(f'{img_name} - No GT available for evaluation.')


def calculate_psnr_ssim(sr_img, gt_img, crop_border, input_order='HWC'):
    '''Calculate PSNR and SSIM between the super-resolved and ground truth images.'''
    # Ensure the images are in correct shape/order
    if input_order == 'HWC':
        sr_img = sr_img.transpose(2, 0, 1)
        gt_img = gt_img.transpose(2, 0, 1)

    # Apply cropping
    if crop_border > 0:
        sr_img = sr_img[:, crop_border:-crop_border, crop_border:-crop_border]
        gt_img = gt_img[:, crop_border:-crop_border, crop_border:-crop_border]

    psnr = calculate_psnr(sr_img, gt_img)
    ssim = calculate_ssim(sr_img, gt_img)
    return psnr, ssim


if __name__ == '__main__':
    opt = parse()  # Parse options from YAML
    test_pipeline(opt)
