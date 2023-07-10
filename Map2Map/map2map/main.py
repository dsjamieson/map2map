from .args import get_args
from . import train
from . import test
import traceback

def main():

    args = get_args()

    if args.mode == 'train':
        train.node_worker(args)
    elif args.mode == 'test':
        test.test(args)


if __name__ == '__main__':
    main()
