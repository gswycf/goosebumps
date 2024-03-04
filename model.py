import torch
import torch.nn as nn

class GooseModel(nn.Module):
    def __init__(self, input_size=128, num_layer=2):
        super(GooseModel, self).__init__()
        self.input_size = input_size
        self.num_layer=num_layer
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=256,
                            num_layers=self.num_layer,
                            batch_first=True)

        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=256)
        self.conv = nn.Conv1d(1, 256)
        self.fc = nn.Linear(256, 2)

    def shift_label(self, y):
        b, l, dim = y.shape
        strat_index = torch.zeros((b,1,dim))
        y = torch.stack([strat_index, y], dim=1)
        return y[b,:-1, :]

    def forward(self, x, y):
        y = torch.sum(y, dim=-1)
        # x [b, 5], y [b,5,4]-> dim
        y = self.shift_label(y)
        y = self.embedding(y) # b 5 4 dim
        x = self.conv(x)  # b 5 dim
        x = self.lstm(X)
        x = x+y


if __name__ == '__main__':
    data=[1,2,3]
    print(data[:-1])
