import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, rnn_type, hidden_size, embedding, n_layers=1, dropout=0, bidirectional=False):
        """

        Args:
            rnn_type (str):
            hidden_size:
            embedding:
            n_layers:
            dropout:
            bidirectional:
        """
        super(EncoderRNN, self).__init__()
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        if rnn_type.lower() == 'lstm':
            self.rnn_layer = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                                     num_layers=n_layers, dropout=(0. if n_layers == 1 else dropout),
                                     bidirectional=bidirectional)
        elif rnn_type.lower() == 'gru':
            self.rnn_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size,
                                    num_layers=n_layers, dropout=(0. if n_layers == 1 else dropout),
                                    bidirectional=bidirectional)
        elif rnn_type.lower() == 'rnn':
            self.rnn_layer = nn.RNN(input_size=hidden_size, hidden_size=hidden_size,
                                    num_layers=n_layers, dropout=(0. if n_layers == 1 else dropout),
                                    bidirectional=bidirectional)
        else:
            raise ValueError(rnn_type, 'rnn_type not supported. Select one of: lstm, gru, rnn')

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)

        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        outputs, hidden = self.rnn_layer(packed, hidden)

        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden
