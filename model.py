import torch
import torch.nn as nn
from word2vec import word2vec_model
import numpy as np


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_token, vector_dim):
        super(TransformerEncoder, self).__init__()
        self.embedding_weights = torch.FloatTensor(word2vec_model.wv.vectors)
        self.embed_dim = embed_dim
        self.embedding  = nn.Embedding.from_pretrained(self.embedding_weights)
        # for param in self.embedding.parameters():
        #     param.requires_grad = False
        self.bgru = nn.GRU(input_size=1500, hidden_size=vector_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.fc = nn.Linear(vector_dim*2, vector_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, input_data, mask):
        batch_size, max_line, max_token = input_data.shape
        embedded_data = self.embedding(input_data.reshape(batch_size, -1))
        embedded_data = embedded_data.reshape(batch_size, max_line, max_token, -1)
        embedded_data = embedded_data.masked_fill(mask.unsqueeze(-1)==False, 0.0)
        bgru_output, _ = self.bgru(embedded_data.reshape(batch_size, max_line, -1))
        output = self.fc(self.dropout(bgru_output))
        
        return output
    
    
class AttentionLayer(nn.Module):
    def __init__(self, sentdim, hidden_size):
        super(AttentionLayer, self).__init__()
        self.W_v = nn.Linear(sentdim, hidden_size)

    def forward(self, x, pdg): # pdg [batch, 100, 100]

        v = self.W_v(x)
        attention_weights = pdg
        v = torch.matmul(attention_weights.unsqueeze(1), v) # v [batch, 1, 100, x]

        return v


class Multi_AttentionLayer(nn.Module):
    def __init__(self, num_head, hidden_size) -> None:
        super(Multi_AttentionLayer, self).__init__()
        dim = hidden_size // num_head
        self.attention = nn.ModuleList([AttentionLayer(hidden_size, dim) for idx in range(num_head)])
        self.linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, pdg):
        x = torch.cat([attention(x, pdg) for attention in self.attention], -1)
        x = self.linear(x)
        return x

class DFMcnn(nn.Module):
    def __init__(self, in_channel=4, channel_size=128, out_channel=256, class_num=2):
        super(DFMcnn, self).__init__()

        self.single = nn.Conv2d(in_channel, channel_size, (1, sent_dim))
        self.dfms = nn.ModuleList([DFM(channel_size) for _ in range(3)])
        self.tail  = nn.Conv2d(channel_size, out_channel, (1, 1))
        self.relu = nn.ReLU()
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(out_channel, class_num)
        self.drop = nn.Dropout(0.5)

    def forward(self, x, pdg):
        x = x.float()
        x = self.single(x)
        x = self.relu(x)
        
        for dfm in self.dfms:
            x = dfm(x, pdg)
    
        x = self.tail(x)
        r = self.max(x).squeeze()
        r = self.fc(self.drop(r))
        print(r)
        return x


class DFM(nn.Module):
    def __init__(self, channel_size):
        super(DFM, self).__init__()
        self.conv2 = nn.Conv2d(channel_size, channel_size, (3, 1), padding=(1, 0))
        self.linear = nn.Linear(channel_size, channel_size)
        self.conv4 = nn.Conv2d(channel_size*4, channel_size, (1, 1))
        self.multi_attention = Multi_AttentionLayer(num_head=2, hidden_size=channel_size)
        self.relu = nn.ReLU()

    def forward(self, x, pdg):
        
        r = x
        x = self.conv2(x)
        x = self.relu(x)
        
        x = x.permute(0, -1, 2, 1) 
        
        rev = []
        for i in range(4):
            rev.append(self.multi_attention(x, pdg[:, i]).permute(0, -1, 2, 1))
        
        x = torch.cat(rev, 1)
       
        x = self.conv4(x)
        x = r + x
        x = self.relu(x)
        return x
    


class CombinedModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_token, sent_dim):
        super(CombinedModel, self).__init__()
        self.transformer_model = TransformerEncoder(vocab_size, embed_dim, max_token, sent_dim)
        self.dfm_model = DFMcnn()
        self.multi_attention = Multi_AttentionLayer(num_head=4, hidden_size=sent_dim)
        self.linear = nn.Linear(sent_dim, sent_dim)
        self.dropout_probability = 0.5
        
    def multi_channel(self, x, pdg):
        rev = []
    
        for i in range(4):
            _ = self.multi_attention(x.unsqueeze(1), pdg[:, i])
            rev.append(_)
            
        x = torch.cat(rev, 1)
        return x

    def forward(self, input, pdg, train=True):
        mask = input != 0
        drop = torch.rand(input.size()) > self.dropout_probability
        if train:
            input = input * drop.cuda()

        pdg = pdg.float()
        output = self.transformer_model(input, mask)
        output = self.multi_channel(output, pdg)
        output = self.dfm_model(output, pdg)

        return output


vocab_size = len(word2vec_model.wv.key_to_index)
embed_dim = 50
sent_dim = 200
channel_size = 64
max_token = 30
torch.manual_seed(14)
np.random.seed(14)

combined = CombinedModel(vocab_size, embed_dim, max_token, sent_dim)
