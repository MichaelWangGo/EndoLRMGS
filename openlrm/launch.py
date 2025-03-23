import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse

from openlrm.runners import REGISTRY_RUNNERS


def main():

    parser = argparse.ArgumentParser(description='OpenLRM launcher')
    parser.add_argument('runner', type=str, help='Runner to launch')
    parser.add_argument('--freeze_endo_gaussian', action='store_true', help='Freeze EndoGaussian training')
    parser.add_argument('--no-freeze_endo_gaussian', dest='freeze_endo_gaussian', action='store_false', help='Do not freeze EndoGaussian training')
    parser.add_argument('--gaussian_config', type=str, default='/workspace/EndoLRM2/EndoGaussian/arguments/endonerf/cutting_tissues_twice.py', 
                       help='Path to EndoGaussian config file')
    args, unknown = parser.parse_known_args()
    if args.runner not in REGISTRY_RUNNERS:
        raise ValueError('Runner {} not found'.format(args.runner))
    RunnerClass = REGISTRY_RUNNERS[args.runner]
    with RunnerClass(freeze_endo_gaussian=args.freeze_endo_gaussian, gaussian_config=args.gaussian_config) as runner:
        runner.run()


if __name__ == '__main__':

    main()
