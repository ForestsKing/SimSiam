import argparse

import torch

from exp.classify import ExpClassify
from utils.seed import setSeed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--encoder', type=int, default=1)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--stop_grad', type=int, default=1)

    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--save_path', type=str, default='./checkpoints/')
    parser.add_argument('--download', type=bool, default=False)
    parser.add_argument('--train_ratio', type=float, default=0.001)

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--devices', type=int, default=0)

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('\n===================== Args ========================')
    print(args)
    print('===================================================\n')

    setSeed(args.random_seed)

    setting = "Encoder_{0}_Freeze_{1}_Pretrain_{2}_Stopgrad_{3}".format(
        int(args.encoder), int(args.freeze), int(args.pretrain), int(args.stop_grad)
    )

    print(setting)

    exp = ExpClassify(args, setting)
    exp.train()
    exp.test()

    print('Done!')
