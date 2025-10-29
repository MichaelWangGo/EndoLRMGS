import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse

from openlrm.runners import REGISTRY_RUNNERS

def main():
    parser = argparse.ArgumentParser(description='OpenLRM launcher')
    parser.add_argument('runner', type=str, help='Runner to launch')
    parser.add_argument('--freeze_gaussian', action='store_true', help='Freeze FMGaussian training')
    parser.add_argument('--no-freeze_gaussian', dest='freeze_gaussian', action='store_false', help='Do not freeze FMGaussian training')
    parser.add_argument('--gaussian_config', type=str, default=None, help='Path to FMGaussian config file')
    args, unknown = parser.parse_known_args()
    if args.runner not in REGISTRY_RUNNERS:
        raise ValueError('Runner {} not found'.format(args.runner))
    RunnerClass = REGISTRY_RUNNERS[args.runner]
    with RunnerClass(freeze_gaussian=args.freeze_gaussian, gaussian_config=args.gaussian_config) as runner:
        runner.run()

if __name__ == '__main__':
    main()
