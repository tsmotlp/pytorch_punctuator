import torch.nn as nn
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import cv2


class LstmPunctuator(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 hidden_size, num_layers, bidirectional,
                 num_class):
        super(LstmPunctuator, self).__init__()
        # Hyper-parameters
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_class = num_class
        # Components
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            batch_first=True,
                            bidirectional=bool(bidirectional))
        fc_in_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_in_dim, num_class)

    def forward(self, x):
        """
        Args:
            padded_input: N x T
            input_lengths: N
        Returns:
            score: N x T x C
        """
        # Embedding Layer
        x = self.embedding(x)  # N x T x D
        x, _ = self.lstm(x)
        # Output Layer
        score = self.fc(x)
        return score


if __name__ == '__main__':
    model = LstmPunctuator(10, 3, 4, 2, True, 5)
    print(model)

    inputs = torch.randint(10, (3, 6)).long()  # 生成shape为（3，6）的一个torch tensor，tensor中所有的数值在[0, 10)之间
    targets = torch.randint(5, (3, 6)).long()

    inputs[-1, -2:] = 0
    targets[-1, -2:] = -1

    score = model(inputs)
    print('inputs shape', inputs.shape)
    print('targets shape', targets.shape)
    print('score shape', score.shape)

    score = score.view(-1, score.size(-1))
    targets = targets.view(-1)
    loss = torch.nn.functional.cross_entropy(score, targets, ignore_index=-1, reduction='elementwise_mean')
    print('loss:', loss.item())
    _, out = score.topk(1, 1, True)
    out = out.view(1, -1)[0]
    print(out)
    print(targets)
    confusion_matrix = metrics.confusion_matrix(targets.detach().numpy(), out.detach().numpy())
    print(confusion_matrix)
    plt.imsave('./confusion_matrix.png', confusion_matrix, dpi=200, bbox_inches='tight')


