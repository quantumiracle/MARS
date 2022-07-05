import argparse

def get_parser_args():
    parser = argparse.ArgumentParser(description='Arguments of the general launching script for MARS.')

    parser.add_argument('--env', type=str, default=None, help='environment')
    parser.add_argument('--method', dest='marl_method', type=str, default=None, help='method name')
    parser.add_argument('--save_id', type=str, default='0', help='identification number for each run')
    parser.add_argument('--render', type=bool, default=False, help='render the scene')
    parser.add_argument('--seed', type=str, default='random', help='random seed')
    parser.add_argument('--record_video', type=bool, default=False, help='whether recording the video')
    parser_args = parser.parse_args()

    return parser_args