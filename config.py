import argparse


def get_config():
    parser = argparse.ArgumentParser(description="remote_sensing_scene_classification")

    parser.add_argument('--data_path', default='F:/AID_dataset/AID/')
    parser.add_argument('--txt_path', default='./data.txt')
    parser.add_argument('--split_rate', default=[[0, .8], [.8, .9], [.9, 1]])
    parser.add_argument('--class_num', default=30)
    parser.add_argument('--conv_block_count', default=3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epoch_count', default=30)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--loss_function', default='CrossEntropyLoss')
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--lr', default=1e-3)
    return parser.parse_args()
