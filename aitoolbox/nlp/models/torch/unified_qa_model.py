import torch
import torch.nn as nn
import torch.nn.functional as F

from aitoolbox.torchtrain.model import TTModel


class UnifiedQABasicRNN(TTModel):
    def __init__(self, hidden_size, output_size, embedding_dim, vocab_size, ctx_n_layers=1, qus_n_layers=1, dropout=0.):
        super().__init__()
        self.ctx_n_layers = ctx_n_layers
        self.qus_n_layers = qus_n_layers
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)

        self.ctx_gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size,
                              num_layers=ctx_n_layers, dropout=(0. if ctx_n_layers == 1 else dropout),
                              bidirectional=True,
                              batch_first=True)

        self.qus_gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size,
                              num_layers=qus_n_layers, dropout=(0. if qus_n_layers == 1 else dropout),
                              bidirectional=True,
                              batch_first=True)

        self.hidden_1 = nn.Linear(hidden_size*2, hidden_size*2)
        self.out_start_span = nn.Linear(hidden_size*2, output_size)
        self.out_end_span = nn.Linear(hidden_size * 2, output_size)

    def forward(self, context_input, question_input, context_input_lengths, question_input_lengths,
                context_hidden=None, question_hidden=None):

        context_input_lengths, perm_index = context_input_lengths.sort(0, descending=True)
        context_input = context_input[perm_index]

        # Convert word indexes to embeddings
        context_embedded = self.embedding(context_input)
        context_packed = nn.utils.rnn.pack_padded_sequence(context_embedded, list(context_input_lengths), batch_first=True)
        context_outputs, context_hidden = self.ctx_gru(context_packed, context_hidden)
        context_outputs, _ = nn.utils.rnn.pad_packed_sequence(context_outputs, batch_first=True)

        odx = perm_index.view(-1, 1).unsqueeze(1).expand(context_outputs.size(0), context_outputs.size(1), context_outputs.size(2))
        context_outputs = context_outputs.gather(0, odx)


        question_input_lengths, perm_index_q = question_input_lengths.sort(0, descending=True)
        question_input = question_input[perm_index_q]

        question_embedded = self.embedding(question_input)
        question_packed = nn.utils.rnn.pack_padded_sequence(question_embedded, list(question_input_lengths), batch_first=True)
        question_outputs, question_hidden = self.qus_gru(question_packed, question_hidden)
        question_outputs, _ = nn.utils.rnn.pad_packed_sequence(question_outputs, batch_first=True)

        odx = perm_index_q.view(-1, 1).unsqueeze(1).expand(question_outputs.size(0), question_outputs.size(1), question_outputs.size(2))
        question_outputs = question_outputs.gather(0, odx)

        # Sum bidirectional GRU outputs
        context_outputs = context_outputs[:, :, :self.hidden_size] + context_outputs[:, :, self.hidden_size:]
        question_outputs = question_outputs[:, :, :self.hidden_size] + question_outputs[:, :, self.hidden_size:]

        context_outputs = context_outputs[:, context_outputs.size(1)-1]
        question_outputs = question_outputs[:, question_outputs.size(1)-1]

        representation_combo = torch.cat([context_outputs, question_outputs], -1)
        h = self.hidden_1(representation_combo)

        output_start_span = self.out_start_span(h)
        output_start_span = F.softmax(output_start_span, dim=1)

        output_end_span = self.out_end_span(h)
        output_end_span = F.softmax(output_end_span, dim=1)

        # Return output and final hidden state
        return output_start_span, output_end_span # , context_hidden, question_hidden

    def get_loss(self, batch_data, criterion, device):
        paragraph_batch, paragraph_lengths, question_batch, question_lengths, span = batch_data

        paragraph_batch = paragraph_batch.to(device)
        paragraph_lengths = paragraph_lengths.to(device)
        question_batch = question_batch.to(device)
        question_lengths = question_lengths.to(device)
        span = span.to(device)

        output_start_span, output_end_span = self(paragraph_batch, question_batch, paragraph_lengths, question_lengths)

        loss1 = criterion(output_start_span, span[:, 0].long())
        loss2 = criterion(output_end_span, span[:, 1].long())
        loss = loss1 + loss2

        return loss

    def get_predictions(self, batch_data, device):
        paragraph_batch, paragraph_lengths, question_batch, question_lengths, span = batch_data

        paragraph_batch = paragraph_batch.to(device)
        paragraph_lengths = paragraph_lengths.to(device)
        question_batch = question_batch.to(device)
        question_lengths = question_lengths.to(device)

        output_start_span, output_end_span = self(paragraph_batch, question_batch, paragraph_lengths, question_lengths)

        _, output_start_span_idx = output_start_span.max(1)
        _, output_end_span_idx = output_end_span.max(1)

        y_test = span
        y_pred = torch.stack((output_start_span_idx, output_end_span_idx), 1)

        metadata = None

        return y_pred.cpu(), y_test.cpu(), metadata
