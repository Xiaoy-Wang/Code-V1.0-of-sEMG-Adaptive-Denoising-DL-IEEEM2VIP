import torch
import torch.nn as nn
from models.SCAttention import SCAttBlock

# 输入：batch, 1, 9, 128
# 输出：batch, num_classes

num_classes = 5


class CNNTransformer(nn.Module):
    def __init__(self, transformer_average):
        super(CNNTransformer, self).__init__()
        self.transformer_average = transformer_average
        assert self.transformer_average in ['time_average', 'channel_average']
        self.model_name = 'CNNTransformer-' + self.transformer_average

        self.cnn_part1_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 5), stride=(1, 1),
                                                   padding='same'),
                                         nn.BatchNorm2d(16),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
                                         # window: 64/128/192/256, kernel_size/stride:(1, 2)/(1, 4)/(1, 6)/(1, 8)
                                         nn.Dropout(p=0.2),
                                         )

        self.cnn_attention = SCAttBlock(op_channel=16, group_kernel_size=(1, 3))

        self.cnn_part1_2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 1),
                                                   padding='same'),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
                                         nn.Dropout(p=0.2),

                                         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1),
                                                   padding='same'),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8)),
                                         nn.Dropout(p=0.2),
                                         )

        self.cnn_part2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 1), stride=(1, 1),
                                                 padding='same'),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
                                       nn.Dropout(p=0.2),

                                       nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 1), stride=(1, 1),
                                                 padding='same'),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.AvgPool2d(kernel_size=(3, 1), stride=(1, 1)),
                                       nn.Dropout(p=0.2),
                                       )

        ################ TRANSFORMER BLOCK #############################
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=16 * 9,  # input feature dim
            nhead=4,  # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            dim_feedforward=256,  # 2 linear layers in each encoder block's feedforward network: dim 32-->256--->32
            dropout=0.2,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)

        linear_in_features = 256 + 32 if self.transformer_average == 'time_average' else 256 + 16 * 9
        self.linear_part = nn.Sequential(nn.Linear(in_features=linear_in_features, out_features=32),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.2),

                                         nn.Linear(in_features=32, out_features=num_classes),
                                         )
        self.init_params()

    def forward(self, data):
        # data: [32, 1, 9, 128]
        batch_size = data.shape[0]
        # 并行分支1： CNN
        cnn1_1_out = self.cnn_part1_1(data)
        cnn_att_out = self.cnn_attention(cnn1_1_out)
        cnn_out = self.cnn_part1_2(cnn_att_out)
        cnn_out = self.cnn_part2(cnn_out)
        branch1_out = cnn_out.reshape(batch_size, -1)
        # print(branch1_out.shape)
        # 并行分支2： Transformer

        data_transformer = cnn_att_out.permute(0, 3, 2, 1).contiguous().view(cnn_att_out.size(0), cnn_att_out.size(-1), -1)
        transformer_output = self.transformer_encoder(data_transformer)
        if self.transformer_average == 'time_average':
            branch2_out = torch.mean(transformer_output, dim=2)  # dim 32x144 --> 32
        else:
            branch2_out = torch.mean(transformer_output, dim=1)  # dim 32x144 --> 144
        # print(branch2_out.shape)
        # 拼接
        branchs_out = torch.cat((branch1_out, branch2_out), dim=1)
        # print(branchs_out.size())

        # 输出层
        out = self.linear_part(branchs_out)

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

    def get_model_name(self):
        return self.model_name
