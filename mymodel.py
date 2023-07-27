import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class DFM_BLOCK(nn.Module):
    def __init__(self, channel_size) -> None:
        super(DFM_BLOCK, self).__init__()
        self.channel_size = channel_size
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), padding=(1, 0))
        self.fuse1 = nn.Conv2d(self.channel_size * 2, self.channel_size, (1, 1))
        self.fuse2 = nn.Conv2d(self.channel_size * 4, self.channel_size, (1, 1))
        self.linear = nn.Linear(1, 1)
        self.relu = nn.ReLU()

    def forward(self, x, pdg):
        single = x
        x = self.conv3(x)
        x = torch.cat([single, x], 1)
        x = self.fuse1(x)
        res = x = self.relu(x)
        x = torch.cat([torch.matmul(pdg, x[:, i:i + 1, :, :]) for i in range(x.size(1))], 1)
        x = self.fuse2(x)
        x = res + x
        x = self.relu(x)
        return x


class DFMCNN(nn.Module):
    def __init__(self) -> None:
        super(DFMCNN, self).__init__()

        self.in_channel = 4
        self.channel_size = 512
        self.out_channel = 1024
        self.class_num = 2

        self.single_line_conv = nn.Conv2d(self.in_channel, self.channel_size, (1, 200))
        self.dfm1 = DFM_BLOCK(512)
        self.dfm2 = DFM_BLOCK(512)
        self.tail = nn.Conv2d(self.channel_size, self.out_channel, (1, 1))
        self.relu = nn.ReLU()
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(self.out_channel, self.class_num)
        self.drop = nn.Dropout(0.5)

    def channel(self, x):
        inp = x.float()
        batchlen = inp.size()[0]
        code = inp[:, 0][:, :125, :]
        code = code.reshape(batchlen, 1, 125, 1, 200)
        pdg = torch.zeros(batchlen, 4, 125, 125)
        cdg_in = inp[:, 1][:, :125, :125]
        cdg_out = torch.transpose_copy(cdg_in, 1, 2)
        ddg_in = inp[:, 2][:, :125, :125]
        ddg_out = torch.transpose_copy(ddg_in, 1, 2)
        pdg[:, 0] = cdg_in
        pdg[:, 1] = cdg_out
        pdg[:, 2] = ddg_in
        pdg[:, 3] = ddg_out
        pdg = pdg.to(self.device)
        code = self.linear(code)
        out = torch.matmul(pdg, code.squeeze(3))

        return out, pdg

    def visulize(self, x, shape, index):
        x = torch.argmax(x, dim=2).cpu().numpy()
        temp = [0 for i in range(shape)]
        temp2 = [0 for i in range(shape)]
        for i in range(self.out_channel):
            temp[x[0, i, 0]] += 1
        for i in range(len(index)):
            temp2[i] = int(index[i])

        sum_dict = {}

        for num, idx in zip(temp, temp2):
            if idx in sum_dict:
                sum_dict[idx] += num
            else:
                sum_dict[idx] = num

        sorted_dict = {k: v for k, v in sorted(sum_dict.items())}
        temp = list(sorted_dict.values())
        temp2 = list(sorted_dict.keys())
        plt.imshow(np.expand_dims(temp, axis=-1), cmap='hot')
        plt.yticks(range(len(temp)), range(1, len(temp) + 1))
        plt.xticks([])
        plt.savefig('heatmap.pdf', format='pdf')
        plt.show()
        # you can use temp and temp2 to map source code
        # temp
        return temp, temp2

    def forward(self, x, index):
        x, pdg = self.channel(x)
        shape = x.shape[2]
        x = self.single_line_conv(x)
        x = self.dfm1(x, pdg)
        x = self.dfm2(x, pdg)
        v = x = self.tail(x)
        x = self.max(x).squeeze()
        x = self.fc(self.drop(x))
        # if x[-1] > x[0]:
        #     self.visulize(x, shape, index)
        return x
