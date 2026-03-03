import argparse
from pathlib import Path
import torch

from src import config
from src.common import setup_seed


def main():
    parser = argparse.ArgumentParser(
        description='Arguments for running StructRecon.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument(
        '--input_folder',
        type=str,
        help='Input folder. This overrides the value in the config file when provided.',
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output folder. This overrides the value in the config file when provided.',
    )
    parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU support')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument(
        '--debug_mode',
        type=str,
        choices=['all', 'mapper', 'tracker'],
        default='all',
        help='Debug mode: all-normal run, mapper-only, tracker-only',
    )
    parser.add_argument(
        '--debug_port',
        type=int,
        default=5678,
        help='Remote debug port for the VSCode debugger',
    )
    args = parser.parse_args()

    default_config = Path(__file__).resolve().parent / 'configs' / 'structrecon.yaml'
    cfg = config.load_config(args.config, str(default_config))

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    seed = args.seed if args.seed is not None else cfg.get('seed', 3407)
    setup_seed(seed)

    if args.multi_gpu:
        if torch.cuda.device_count() > 1:
            cfg['use_multi_gpu'] = True
            print(f"Enabling multi-GPU support with {torch.cuda.device_count()} GPUs")
        else:
            print('Warning: --multi_gpu was set, but only one GPU is available. Running in single-GPU mode.')

    if args.debug_mode != 'all':
        print(f"Debug mode enabled: {args.debug_mode}, debug port: {args.debug_port}")
        cfg['debug'] = {
            'mode': args.debug_mode,
            'port': args.debug_port,
        }

    from src.StructRecon import StructRecon
    structrecon = StructRecon(cfg, args)
    structrecon.run()


if __name__ == '__main__':
    main()
