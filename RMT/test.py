"""
Test script for model evaluation
Usage: python test.py --resume /path/to/checkpoint.pth --data-path /path/to/test_dataset
"""

import argparse
import os
import json
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from timm.utils import ModelEma
from classfication_release.datasets import build_dataset
from classfication_release.engine import evaluate
from classfication_release import utils
from classfication_release.RMT import RMT_T3, RMT_S, RMT_M2, RMT_L6
from classfication_release.RMT_LA import RMT_S_LA
from classfication_release.RMT_SA import RMT_S_SA
from classfication_release.RMT_CBAM import RMT_S_CBAM
from classfication_release.RMT_PSCA import RMT_S_PSCA
from classfication_release.RMT_FSA import RMT_S_FSA
from classfication_release.RMT_GCSA import RMT_S_GCSA
from classfication_release.RMT_CA import RMT_S_CA

# Model architecture mapping
archs = {
    'RMT_T': RMT_T3,
    'RMT_S': RMT_S,
    'RMT_B': RMT_M2,
    'RMT_L': RMT_L6,
    'RMT_S_CA': RMT_S_CA,
    'RMT_S_LA': RMT_S_LA,
    'RMT_S_SA': RMT_S_SA,
    'RMT_S_CBAM': RMT_S_CBAM,
    'RMT_S_PSCA': RMT_S_PSCA,
    'RMT_S_FSA': RMT_S_FSA,
    'RMT_S_GCSA': RMT_S_GCSA
}


def get_args_parser():
    parser = argparse.ArgumentParser('Model testing script', add_help=False)

    # Required parameters
    parser.add_argument('--resume', default=r'C:\Users\Administrator\Desktop\RMT-main\save\LA\checkpoint.pth', type=str, help='path to checkpoint')
    parser.add_argument('--data-path', default=r'C:\Users\Administrator\Desktop\38044', type=str, help='dataset path')

    # Model parameters
    parser.add_argument('--model', default='RMT_S_LA', type=str,
                        help='Name of model to test')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    # Dataset parameters
    parser.add_argument('--data-set', default='a', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'a'],
                        type=str, help='Image Net dataset type')

    # Device parameters
    parser.add_argument('--device', default='cuda', help='device to use for testing')
    parser.add_argument('--num-workers', default=4, type=int, help='number of data loading workers')

    # Distributed testing (optional)
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # Output options
    parser.add_argument('--output-dir', default=r'C:\Users\Administrator\Desktop\RMT-main\results\LA', help='path where to save results')

    return parser


def load_model(args):
    """Load model from checkpoint"""
    print(f"Creating model: {args.model}")
    model = archs[args.model](args)

    # Load checkpoint
    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu')
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Load state dict
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {args.resume}")
    print(f"Missing keys: {msg.missing_keys}")
    print(f"Unexpected keys: {msg.unexpected_keys}")

    # Handle EMA if present
    model_ema = None
    if 'model_ema' in checkpoint:
        model_ema = ModelEma(model, decay=0)
        model_ema.ema.load_state_dict(checkpoint['model_ema'])
        print("Loaded EMA model")

    return model, model_ema


def main(args):
    # Initialize distributed mode (if needed)
    utils.init_distributed_mode(args)

    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Fix random seeds for reproducibility
    # torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Prepare dataset
    print(f"Loading test dataset from {args.data_path}")
    dataset_test, _ = build_dataset(is_train=False, args=args)

    # Create data loader
    if args.distributed:
        sampler_test = torch.utils.data.DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=128,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Load model
    model, model_ema = load_model(args)
    model.to(args.device)
    model.eval()
    if model_ema is not None:
        model_ema.ema.to(args.device)

    # Print model info
    print(f"Model: {args.model}")
    print(f"Number of test images: {len(dataset_test)}")

    # Evaluate model
    print("Starting evaluation...")
    test_stats = evaluate(data_loader_test, model, args.device)
    print(f"Accuracy of the network on {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")

    # Evaluate EMA model if exists
    if model_ema is not None:
        test_stats_ema = evaluate(data_loader_test, model_ema.ema, args.device)
        print(f"Accuracy of the EMA network: {test_stats_ema['acc1']:.1f}%")

    # Save results
    if args.output_dir and utils.is_main_process():
        results = {
            'model': args.model,
            'dataset': args.data_path,
            'test_stats': test_stats,
            'test_stats_ema': test_stats_ema if model_ema else None
        }

        with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model testing script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)