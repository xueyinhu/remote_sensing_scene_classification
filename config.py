import argparse


def get_config():
    parser = argparse.ArgumentParser(description="remote_sensing_scene_classification")
    parser.add_argument('--data_path', default='F:/NWPU-RESISC45/')
    parser.add_argument('--txt_path', default='./data.txt')
    parser.add_argument('--split_rate', default=[[0, 0.8], [0.8, 0.9], [0.9, 1]])
    parser.add_argument('--class_num', default=45)
    parser.add_argument('--epoch_count', default=30)
    parser.add_argument('--batch_size', default=128)
    return parser.parse_args()
