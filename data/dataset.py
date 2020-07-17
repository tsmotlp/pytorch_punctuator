import torch
import numpy as np
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, data_path, max_num_words=None, name='train'):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.max_num_words = max_num_words
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inputs, self.labels = self.tensorFromSamples()

    def __getitem__(self, index):
        input, label = self.inputs[index], self.labels[index]
        return {'inputs':input[:self.max_num_words], 'labels':label[:self.max_num_words]}

    def __len__(self):
        return len(self.inputs)

    def loadSamples(self):
        samples = np.load(self.data_path, allow_pickle=True)
        return samples

    def tensorFromSamples(self):
        print("Loading {} data from disk...".format(self.name))
        samples = self.loadSamples()
        inputs, labels = list(samples[0]), list(samples[1])
        # print(len(inputs), len(labels))
        print("Transfer {} data to Tensor...".format(self.name))
        inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        print("Transfer complete~")
        return inputs_tensor, labels_tensor


def get_dataloader(args):
    if args.mode == 'train':
        train_dataset = Dataset(args.train_data_path, args.max_num_words, name='train')
        valid_dataset = Dataset(args.valid_data_path, args.max_num_words, name='valid')

        train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                           num_workers=args.nThreads, drop_last=True)
        valid_dataloader = data.DataLoader(dataset=valid_dataset, batch_size=args.valid_batch_size, shuffle=False,
                                           num_workers=args.nThreads, drop_last=True)

        return train_dataloader, valid_dataloader

    else:
        test_dataset = Dataset(args.test_data_path, args.max_num_words, name='test')
        test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                          num_workers=args.nThreads, drop_last=True)
        return test_dataloader

if __name__ == '__main__':
    ds = Dataset('./processed_data/train.npy')
    print(ds.__len__())
    dl = data.DataLoader(dataset=ds, batch_size=10, shuffle=True, num_workers=0)
    for i, (inputs, labels) in enumerate(dl):
        print(inputs.shape, labels.shape)