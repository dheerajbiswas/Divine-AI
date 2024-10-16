import os
import torch
from collections import OrderedDict
from mmsr.models.archs import domain_matching  # Import Domain Matching module
from mmsr.models import build_model
from mmsr.utils import get_root_logger, make_exp_dirs, parse_options, dict2str
from mmsr.data import create_dataloader, create_dataset
from mmsr.metrics import calculate_psnr_ssim  # Assuming a metrics file exists for evaluation

def train_pipeline(opt):
    '''Main training pipeline to run C2-matching with Domain Matching preprocessing'''

    # Set up environment and logging
    torch.backends.cudnn.benchmark = True
    logger = get_root_logger()
    logger.info('Set log level to {}'.format(logger.level))
    logger.info(dict2str(opt))

    # Create experiment directories
    make_exp_dirs(opt)

    # Create training and validation datasets and dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt, phase)
            logger.info(f'Number of training images in {dataset_opt["name"]}: {len(train_set)}')
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, phase)
            logger.info(f'Number of validation images in {dataset_opt["name"]}: {len(val_set)}')

    # Build model
    model = build_model(opt)

    # Initialize Domain Matching model
    domain_matching_model = domain_matching.DomainMatchingSR().to(opt['device'])

    current_iter = 0
    total_iters = int(opt['train']['total_iter'])
    logger.info('Start training from iteration 0')

    while current_iter < total_iters:
        for train_data in train_loader:
            current_iter += 1

            # Send low-resolution (LQ) and reference (Ref) images to device
            model.feed_data(train_data)

            # Apply Domain Matching preprocessing before SR
            lr_img = train_data['LQ'].to(opt['device'])
            ref_img = train_data['ref'].to(opt['device'])

            # Use Domain Matching model to preprocess LQ and Ref
            lr_transformed = domain_matching_model(lr_img, ref_img)

            # Set the preprocessed LQ image back to the model and perform forward pass
            model.feed_data({'LQ': lr_transformed, 'ref': ref_img})
            model.optimize_parameters()

            # Log training progress
            if current_iter % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = f'[Iter: {current_iter}/{total_iters}] '
                for k, v in logs.items():
                    message += f'{k}: {v:.4f} '
                logger.info(message)

            # Save models and validation results
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and state.')
                model.save(current_iter)

            # Perform validation
            if val_loader and current_iter % opt['train']['val_freq'] == 0:
                logger.info('Performing validation.')
                for val_data in val_loader:
                    model.feed_data(val_data)
                    model.test()

                    # You can log validation metrics, save images, etc.
                    visuals = model.get_current_visuals(need_GT=True)
                    sr_img = tensor2img([visuals['rlt']])
                    gt_img = tensor2img([visuals['GT']]) if 'GT' in visuals else None

                    # Calculate validation metrics if needed
                    if gt_img is not None:
                        psnr, ssim = calculate_psnr_ssim(sr_img, gt_img, crop_border=opt['crop_border'], input_order='HWC')
                        logger.info(f'Validation PSNR: {psnr:.6f}, SSIM: {ssim:.6f}')

            if current_iter >= total_iters:
                logger.info('End of training.')
                break

if __name__ == '__main__':
    opt = parse_options(is_train=True)  # Parse options from YAML file
    train_pipeline(opt)
