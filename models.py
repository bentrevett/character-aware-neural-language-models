import torch
import torch.nn as nn
import torch.nn.functional as F

class CharLM(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, filter_sizes, filter_channels, highway_layers, rnn_dim, rnn_layers, dropout):
        super().__init__()

        self.embedding = TimeDistributed(nn.Embedding(input_dim, embedding_dim))

        #putting these all in a list and then doing a list comprehension in the forward method
        #doesn't seem to work, gives weird error. so, for now, am doing it this ugly way.
        self.cnn1 = TimeDistributed(nn.Conv2d(1, filter_channels[0], (filter_sizes[0], embedding_dim)))
        self.cnn2 = TimeDistributed(nn.Conv2d(1, filter_channels[1], (filter_sizes[1], embedding_dim)))
        self.cnn3 = TimeDistributed(nn.Conv2d(1, filter_channels[2], (filter_sizes[2], embedding_dim)))
        self.cnn4 = TimeDistributed(nn.Conv2d(1, filter_channels[3], (filter_sizes[3], embedding_dim)))
        self.cnn5 = TimeDistributed(nn.Conv2d(1, filter_channels[4], (filter_sizes[4], embedding_dim)))
        self.cnn6 = TimeDistributed(nn.Conv2d(1, filter_channels[5], (filter_sizes[5], embedding_dim)))

        self.highway = TimeDistributed(Highway(sum(filter_channels), 1, F.relu))

        self.rnn = nn.LSTM(sum(filter_channels), rnn_dim, rnn_layers, dropout=dropout)

        self.fc = nn.Linear(rnn_dim, output_dim)

    def forward(self, chars):

        #chars = [bsz, seq. len, n_chars]

        embedded_chars = self.embedding(chars)

        #embedded_chars = [bsz, seq len, n_chars, emb dim]

        embedded_chars = embedded_chars.unsqueeze(2) #need to add a dummy 'channel' dimension

        #embedded_chars = [bsz, seq len, 1, n_chars, emb dim]

        cnn1_output = F.tanh(self.cnn1(embedded_chars).squeeze(-1))
        cnn2_output = F.tanh(self.cnn2(embedded_chars).squeeze(-1))
        cnn3_output = F.tanh(self.cnn3(embedded_chars).squeeze(-1))
        cnn4_output = F.tanh(self.cnn4(embedded_chars).squeeze(-1))
        cnn5_output = F.tanh(self.cnn5(embedded_chars).squeeze(-1))
        cnn6_output = F.tanh(self.cnn6(embedded_chars).squeeze(-1))

        #each cnn_output = [bsz, seq len, n_filters[i], n_chars-(filter_width[i]-1)]        

        cnn1_max = F.max_pool2d(cnn1_output, (1, cnn1_output.shape[3])).squeeze(-1) 
        cnn2_max = F.max_pool2d(cnn2_output, (1, cnn2_output.shape[3])).squeeze(-1) 
        cnn3_max = F.max_pool2d(cnn3_output, (1, cnn3_output.shape[3])).squeeze(-1) 
        cnn4_max = F.max_pool2d(cnn4_output, (1, cnn4_output.shape[3])).squeeze(-1) 
        cnn5_max = F.max_pool2d(cnn5_output, (1, cnn5_output.shape[3])).squeeze(-1) 
        cnn6_max = F.max_pool2d(cnn6_output, (1, cnn6_output.shape[3])).squeeze(-1)
        
        #each cnn_max = [bsz, seq len, n_filters[i]]

        cnns_max = torch.cat((cnn1_max, cnn2_max, cnn3_max, cnn4_max, cnn5_max, cnn6_max), dim=2)

        #cnns_max = [bsz, seq len, sum(n_filters)] 

        highway_output = self.highway(cnns_max)

        #highway_output = [bsz, seq len, sum(n_filters)]

        output, (hidden, cell) = self.rnn(highway_output)

        #output = [seq len, bsz, rnn_dim]

        decoded = self.fc(output.view(output.size(0)*output.size(1), output.size(2))) 
        
        #decoded = [seq len * bsz, output_dim]

        return decoded


class TimeDistributed(nn.Module):
    """
    Given an input shaped like ``(batch_size, time_steps, [rest])`` and a ``Module`` that takes
    inputs like ``(batch_size, [rest])``, ``TimeDistributed`` reshapes the input to be
    ``(batch_size * time_steps, [rest])``, applies the contained ``Module``, then reshapes it back.
    Note that while the above gives shapes with ``batch_size`` first, this ``Module`` also works if
    ``batch_size`` is second - we always just combine the first two dimensions, then split them.
    """
    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, *inputs):  # pylint: disable=arguments-differ
        reshaped_inputs = []
        for input_tensor in inputs:
            input_size = input_tensor.size()
            if len(input_size) <= 2:
                raise RuntimeError("No dimension to distribute: " + str(input_size))

            # Squash batch_size and time_steps into a single axis; result has shape
            # (batch_size * time_steps, input_size).
            squashed_shape = [-1] + [x for x in input_size[2:]]
            reshaped_inputs.append(input_tensor.contiguous().view(*squashed_shape))

        reshaped_outputs = self._module(*reshaped_inputs)

        # Now get the output back into the right shape.
        # (batch_size, time_steps, [hidden_size])
        new_shape = [input_size[0], input_size[1]] + [x for x in reshaped_outputs.size()[1:]]
        outputs = reshaped_outputs.contiguous().view(*new_shape)

        return outputs

class Highway(torch.nn.Module):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_ does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.
    This module will apply a fixed number of highway layers to its input, returning the final
    result.
    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of :math:`x`.  We assume the input has shape ``(batch_size,
        input_dim)``.
    num_layers : ``int``, optional (default=``1``)
        The number of highway layers to apply to the input.
    activation : ``Callable[[torch.Tensor], torch.Tensor]``, optional (default=``torch.nn.functional.relu``)
        The non-linearity to use in the highway layers.
    """
    def __init__(self, input_dim, num_layers = 1, activation = torch.nn.functional.relu):
        super(Highway, self).__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim * 2)
                                            for _ in range(num_layers)])
        self._activation = activation
        for layer in self._layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, to we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias[input_dim:].data.fill_(1)


    def forward(self, inputs):
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify the initialization
            # above, too.
            nonlinear_part = projected_input[:, (0 * self._input_dim):(1 * self._input_dim)]
            gate = projected_input[:, (1 * self._input_dim):(2 * self._input_dim)]
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.nn.functional.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input