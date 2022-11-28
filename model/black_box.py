from torch import nn

class MANN(nn.Module):
    def __init__(self, k_shot, n_way, batch_size, img_size ):
        super(MANN, self).__init__()
        self.k_shot = k_shot
        self.n_way = n_way
        self.batch_size = batch_size
        self.img_size = img_size
        self.lstm1 = nn.LSTM(img_size + n_way, 128, batch_first = True)#, bidirectional = bi_dir
        self.lstm2 = nn.LSTM(128, n_way, batch_first = True)
    def forward(self, x):
        x,_ = self.lstm1(x)
        x,_ = self.lstm2(x)
        return x[:,-self.n_way:]
        