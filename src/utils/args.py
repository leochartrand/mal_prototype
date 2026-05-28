import argparse

def parse_args(args):
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--config', '--cfg', type=str, required=True, help='Config file name (relative to config/)')
    parser.add_argument('--gpu', type=int, required=False, help='GPU device ID, or -1 Parallel if using multiple GPUs', default='0')
    parser.add_argument('--frontend', action='store_true', help='Launch the frontend server')
    parser.add_argument('--dummy', action='store_true', help='Use dummy model for debugging purposes')
    
    parsed_args = parser.parse_args(args)
    return parsed_args
