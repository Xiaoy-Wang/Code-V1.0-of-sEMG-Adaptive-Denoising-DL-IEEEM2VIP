import torch.nn as nn

# 输入：batch, 1, 9, 128
# 输出：batch, num_classes

num_classes = 5


class CNNRNNs(nn.Module):
    def __init__(self, rnn_type):
        super(CNNRNNs, self).__init__()
        assert rnn_type in ['LSTM', 'GRU', 'BiLSTM', 'BiGRU']
        self.model_name = 'CNN-'+rnn_type
        # 32, 1, 9, 128 -- 32, 16, 9, 32 -- 32, 32, 9, 8
        self.cnn_part = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 5), stride=(1, 1),
                                                padding='same'),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
                                      nn.Dropout(p=0.2),

                                      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 1),
                                                padding='same'),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
                                      # window: 64/128/192/256, kernel_size/stride:(1, 2)/(1, 4)/(1, 6)/(1, 8)
                                      nn.Dropout(p=0.2),
                                      )
        # 32, 32, 9, 8 -- 32, 8, 9*32  -- 32, 8, 64 / 128
        # 32, 8, 64 / 128 -- 32, 16, 128 / 256 -- 32, 128 / 256
        # RNNs
        if rnn_type == 'LSTM':
            self.rnn_part1 = nn.LSTM(input_size=9 * 32, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
            self.rnn_part2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        elif rnn_type == 'GRU':
            self.rnn_part1 = nn.GRU(input_size=9 * 32, hidden_size=64, num_layers=1, batch_first=True, bidirectional=False)
            self.rnn_part2 = nn.GRU(input_size=64, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        elif rnn_type == 'BiLSTM':
            self.rnn_part1 = nn.LSTM(input_size=9 * 32, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
            self.rnn_part2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        else:
            self.rnn_part1 = nn.GRU(input_size=9 * 32, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
            self.rnn_part2 = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)

        # Classifier: 32, 128 / 256 -- 32, 32 -- 32, 5
        linear_in_features = 128 if rnn_type in ['LSTM', 'GRU'] else 256
        self.linear_part = nn.Sequential(nn.Linear(in_features=linear_in_features, out_features=32),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.2),

                                         nn.Linear(in_features=32, out_features=num_classes),
                                         )
        self.init_params()

    def forward(self, data):

        cnn_out = self.cnn_part(data)
        # print(cnn_out.size())

        # 将CNN的输出形状调整为LSTM的输入形状
        # [32, 32, 9, 16] --  [32, 16, 32, 9] --  [32, 16, 9*32]
        cnn_out = cnn_out.permute(0, 3, 2, 1).contiguous().view(cnn_out.size(0), cnn_out.size(-1), -1)
        # print(cnn_out.size())
        rnn_out, _ = self.rnn_part1(cnn_out)
        # print(rnn_out.size())
        rnn_out, _ = self.rnn_part2(rnn_out)
        # print(rnn_out.size())
        rnn_out = rnn_out[:, -1, :]

        # 输出层
        out = self.linear_part(rnn_out)
        return out

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def get_model_name(self):
        return self.model_name
