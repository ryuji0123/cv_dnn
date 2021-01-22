import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Configuration for poincare embedding')
    parser.add_argument('--cfg_file', type=str, default='./config/sample.yaml')
    args = parser.parse_args()
    return args
