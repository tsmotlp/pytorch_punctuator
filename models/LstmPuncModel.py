import torch
from .BaseModel import BaseModel
from utils import init_net, get_scheduler, Metrics
from .LstmPuncNet import LstmPunctuator


class LstmPuncModel(BaseModel):
    def __init__(self, args):
        BaseModel.__init__(self, args)
        self.metrics = Metrics()    # 一些测量指标
        self.visual_images = []     # 需要可视化的图像
        self.visual_losses = []     # 需要可视化的loss
        if self.args.mode == 'train':
            self.visual_images += ['train_confusion_matrix']
            self.visual_losses += ['train_loss', 'train_precision', 'train_recall', 'train_f1_score']
        if self.args.mode == 'valid':
            self.visual_images += ['valid_confusion_matrix']
            self.visual_losses += ['valid_loss', 'valid_precision', 'valid_recall', 'valid_f1_score']
        if self.args.mode == 'test':
            self.visual_images += ['test_confusion_matrix']
            self.visual_losses += ['test_loss', 'test_precision', 'test_recall', 'test_f1_score']

    def setup(self):
        if self.args.mode == 'train':
            self.set_random_seed(self.args.seed)
            # 以{'网络名称'：网络对象}的字典的形式申明网络结构（有几个就添加几个）
            self.networks = {'LstmPuncNet': LstmPunctuator(self.args.vocab_size, self.args.embedding_dim, self.args.hidden_size,
                                                           self.args.num_layers, bidirectional=True, num_class=self.args.num_classes)}
            # 网络结构的初始化
            self.networks = init_net(self.networks, self.args.init_type, self.args.gpu_ids)

            # 优化器
            self.optimizers = {'optimizer': torch.optim.SGD(self.networks['LstmPuncNet'].parameters(), lr=self.args.lr, weight_decay=1e-4)}

            # 学习率衰减策略
            self.schedulers = [get_scheduler(optimizer, self.args) for optimizer in list(self.optimizers.values())]

            # 损失函数，可以在这里写一些备用的，方便修改
            self.objectives = {'CrossEntropyLoss': torch.nn.CrossEntropyLoss(ignore_index=self.args.ignore_index).to(self.device),
                               'NLLLoss': torch.nn.NLLLoss().to(self.device)}
        else:
            self.networks = {'LstmPuncNet': LstmPunctuator(self.args.vocab_size, self.args.embedding_dim, self.args.hidden_size,
                                                           self.args.num_layers, bidirectional=True, num_class=self.args.num_classes)}
            self.objectives = {'CrossEntropyLoss': torch.nn.CrossEntropyLoss(ignore_index=self.args.ignore_index).to(self.device),
                               'NLLLoss': torch.nn.NLLLoss().to(self.device)}
            self.load_networks(self.networks, self.args.load_epoch)

    def set_input(self, input):
        """
        设置网络的输入
        """
        self.inputs = input['inputs'].to(self.device)
        self.labels = input['labels'].to(self.device)

    def forward(self):
        """
        这部分写自己的前向计算，loss，metrics

        """
        self.outputs = self.networks['LstmPuncNet'](self.inputs)
        self.outputs = self.outputs.view(-1, self.outputs.size(-1))


        self.labels = self.labels.view(-1)
        self.loss = self.objectives['CrossEntropyLoss'](self.outputs, self.labels)
        if self.args.mode == 'train':
            self.train_precision = self.metrics.precision(self.labels, self.outputs)
            self.train_recall = self.metrics.recall(self.labels, self.outputs)
            self.train_f1_score = self.metrics.f1_score(self.labels, self.outputs)
            self.train_confusion_matrix = self.metrics.confusion_matrix(self.labels, self.outputs)
            self.train_loss = self.loss
            self.metric = self.loss
        if self.args.mode == 'valid':
            self.valid_precision = self.metrics.precision(self.labels, self.outputs)
            self.valid_recall = self.metrics.recall(self.labels, self.outputs)
            self.valid_f1_score = self.metrics.f1_score(self.labels, self.outputs)
            self.valid_confusion_matrix = self.metrics.confusion_matrix(self.labels, self.outputs)
            self.valid_loss = self.loss
        if self.args.mode == 'test':
            self.test_precision = self.metrics.precision(self.labels, self.outputs)
            self.test_recall = self.metrics.recall(self.labels, self.outputs)
            self.test_f1_score = self.metrics.f1_score(self.labels, self.outputs)
            self.test_confusion_matrix = self.metrics.confusion_matrix(self.labels, self.outputs)

    def backward(self):
        """
        反向传播
        """
        # loss
        self.loss.backward()


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        self.optimizers['optimizer'].zero_grad()  # set G_A and G_B's gradients to zero
        self.backward()  # calculate gradients for G_A and G_B
        self.optimizers['optimizer'].step()  # update G_A and G_B's weights


