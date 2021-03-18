import argparse

from os.path import join

from config.const import PROJECT_ROOT


def parse_console():
    parser = argparse.ArgumentParser(description='Configuration for experiments')
    parser.add_argument('--args_file_path', type=str)
    parser.add_argument(
            '--cfg_file_path', type=str, default=join(PROJECT_ROOT, 'config', 'simple_cnn.yaml')
            )
    parser.add_argument('--tmp_results_dir', type=str)
    parser.add_argument('--train_log_file_path', type=str)
    args = parser.parse_args()
    return args
