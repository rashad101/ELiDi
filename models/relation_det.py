import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils.args import get_args
args = get_args()


class RelationPrediction(nn.Module):
    def __init__(self, n_words, emb_dim, h_dim, target_size, lstm_drop=0.3, dropout=0.5, pretrained_emb=None,
                 train_embed=False):
        super(RelationPrediction, self).__init__()
        target_size = target_size
        if args.relation_prediction_mode.upper() == "GRU":
            self.gru = nn.LSTM(input_size=emb_dim,
                               hidden_size=h_dim,
                               num_layers=1,
                               dropout=lstm_drop,
                               bidirectional=True)
        # for attention
        self.W = nn.Linear(h_dim*2, 1)
        self.h_dim = h_dim
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.Tanh() # use relu instead
        self.hidden2tag = nn.Sequential(
                nn.Linear(h_dim*2, h_dim*2 ),
                nn.BatchNorm1d(h_dim*2 ),
                self.relu,
                self.dropout,
                nn.Linear(h_dim*2, target_size)
        )

        torch.nn.init.xavier_uniform_(self.W.weight)
        if args.relation_prediction_mode.upper() == "CNN":
            input_channel = 1
            Ks = 100
            output_channel = h_dim // 3
            self.conv1 = nn.Conv2d(input_channel, output_channel, (2, emb_dim), padding=(1, 0))
            self.conv2 = nn.Conv2d(input_channel, output_channel, (3, emb_dim), padding=(2, 0))
            self.conv3 = nn.Conv2d(input_channel, output_channel, (4, emb_dim), padding=(3, 0))
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(Ks * 3, target_size)

    def forward(self, x, m):
        if args.relation_prediction_mode.upper() == "GRU":
            x = self.dropout(x)
            outputs, (ht, ct) = self.gru(x)
            # Attention according to http://www.aclweb.org/anthology/P16-2034
            outputs = outputs.view(x.size(0), x.size(1), 2, self.h_dim)  # S X B X 2 X H
            fw = outputs[:, :, 0, :].squeeze(2) * m.unsqueeze(-1)  # S X B X H
            bw = outputs[:, :, 1, :].squeeze(2) * m.unsqueeze(-1)
            H = torch.cat([fw, bw], dim=-1)  # S X B X H*2
            H = H.transpose(0, 1)  # B X S X H * 2
            M = F.tanh(H)  # B X S X H * 2
            alpha = F.softmax(self.W(M.reshape(-1, self.h_dim*2)).reshape(-1, x.size(0)), dim=-1) * m.transpose(0, 1)  # B X S
            alpha = alpha.unsqueeze(-1)  # B X S X 1
            r = torch.bmm(H.transpose(1, 2), alpha).squeeze(2)  # B X S X 1 ==> B X S
            h_star = F.tanh(r)
            h_drop = self.dropout(h_star)

            scores = self.hidden2tag(h_drop)
            return scores
        elif args.relation_prediction_mode.upper() == "CNN":
            x = x.transpose(0, 1).contiguous().unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
            x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
            # (batch, channel_output, ~=sent_len) * Ks
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
            # (batch, channel_output) * Ks
            x = torch.cat(x, 1)  # (batch, channel_output * Ks)
            x = self.dropout(x)
            logit = F.tanh(self.fc1(x))  # (batch, target_size)
            scores = F.log_softmax(logit, dim=1)
            return scores
        else:
            print("Unknown Mode")
            exit(1)


