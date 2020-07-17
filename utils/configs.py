import argparse


class Configs():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='hyper-parameters')
        self.parser.add_argument('--train_data_path', type=str, default='./data/processed_data/train.npy')
        self.parser.add_argument('--valid_data_path', type=str, default='./data/processed_data/valid.npy')
        self.parser.add_argument('--test_data_path', type=str, default='./data/processed_data/test.npy')
        self.parser.add_argument('--nThreads', type=int, default=0)

        self.parser.add_argument('--num_epochs', type=int, default=100)
        self.parser.add_argument('--save_epoch_freq', type=int, default=1)
        self.parser.add_argument('--valid_epoch_freq', type=int, default=1)
        self.parser.add_argument('--start_epoch', type=int, default=1)

        self.parser.add_argument('--max_num_words', type=int, default=256)
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--train_batch_size', type=int, default=128)
        self.parser.add_argument('--valid_batch_size', type=int, default=128)
        self.parser.add_argument('--hidden_size', type=int, default=512)
        self.parser.add_argument('--vocab_size', type=int, default=107385+1)
        self.parser.add_argument('--embedding_dim', type=int, default=256)
        self.parser.add_argument('--num_layers', type=int, default=2)
        self.parser.add_argument('--num_classes', type=int, default=8)
        self.parser.add_argument('--ignore_index', type=int, default=0)

        self.parser.add_argument('--mode', required=True)
        self.parser.add_argument('--seed', type=int, default=2333)
        self.parser.add_argument('--init_type', type=str, default='normal')
        self.parser.add_argument('--gpu_ids', type=list, default=[0])
        self.parser.add_argument('--load_epoch', type=int, default=1)
        self.parser.add_argument('--lr_policy', type=str, default='plateau')

        self.parser.add_argument('--save_dir', type=str, default='outputs')


    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt