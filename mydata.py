import numpy
import torch
from torch.utils.data import Dataset, DataLoader


def load_for_model(data_list, max_len, hidden_size, batch_size):
    train_set = TraditionalDataset(data_list[0], data_list[1], max_len, hidden_size)
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_set = TraditionalDataset(data_list[2], data_list[3], max_len, hidden_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader, test_loader


class TraditionalDataset(Dataset):
    def __init__(self, texts, targets, max_len, hidden_size):
        self.texts = texts
        self.targets = targets
        self.max_len = max_len
        self.hidden_size = hidden_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        feature = self.texts[idx]
        target = self.targets[idx]
        vectors = numpy.zeros(shape=(3, self.max_len, self.hidden_size))
        for j in range(3):
            for i in range(min(len(feature[0]), self.max_len)):
                if j == 0:
                    vectors[j][i] = feature[j][i]
                else:
                    for k in range(min(len(feature[1][0]), self.hidden_size)):
                        vectors[j][i][k] = feature[j][i][k]
        return {
            'vector': vectors,
            'targets': torch.tensor(target, dtype=torch.long)
            # 'index': feature[-1]
        }
