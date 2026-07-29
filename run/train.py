import argparse

from .run import run


def get_args():
    parser = argparse.ArgumentParser(
        description='Medical Image Segmentation Training Script')

    parser.add_argument('datasets', type=str,
                        help='Dataset names')

    parser.add_argument('--input', type=str,
                        help='Input data root')
    parser.add_argument('--output', type=str,
                        help='Output result root')

    parser.add_argument('--models', type=str,
                        help='Model names')

    parser.add_argument('--batch_size', type=int,
                        help='Batch Size for training')
    parser.add_argument('--epochs', type=int,
                        help='Epochs for training')
    parser.add_argument('--min_epochs', type=int, default=0,
                        help='Min Epochs')
    parser.add_argument('--start_valid_epoch', type=int, default=1,
                        help='First epoch to run validation')
    parser.add_argument('--valid_interval', type=int, default=1,
                        help='Valid')

    parser.add_argument('--opt', type=str,
                        help='Optimizer name')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')

    parser.add_argument('--sched', type=str,
                        help='Scheduler name')

    return parser.parse_args()


def main(args):
    print(args)
    run(args=args)


if __name__ == '__main__':
    opts = get_args()
    main(opts)
