import torch
import torch.nn as nn
import torch.nn.functional as F

class CharLM(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, filter_sizes, filter_channels, highway_layers, hidden_dim, n_layers):
        super().__init__()

        self.embedding = TimeDistributed(nn.Embedding(input_dim, embedding_dim))

        #putting these all in a list and then doing a list comprehension in the forward method
        #doesn't seem to work, gives weird error. so, for now, am doing it this ugly way.
        self.cnn1 = TimeDistributed(nn.Conv2d(1, filter_channels[0], (filter_sizes[0], embedding_dim)))
        self.cnn2 = TimeDistributed(nn.Conv2d(1, filter_channels[1], (filter_sizes[1], embedding_dim)))
        self.cnn3 = TimeDistributed(nn.Conv2d(1, filter_channels[2], (filter_sizes[2], embedding_dim)))
        self.cnn4 = TimeDistributed(nn.Conv2d(1, filter_channels[3], (filter_sizes[3], embedding_dim)))
        self.cnn5 = TimeDistributed(nn.Conv2d(1, filter_channels[4], (filter_sizes[4], embedding_dim)))

        #self.cnn_max = TimeDistributed(nn.MaxPool2d())

    def forward(self, chars):

        print(chars.shape)

        #chars = [bsz, seq. len, n_chars]

        embedded_chars = self.embedding(chars)

        print(embedded_chars.shape)

        #embedded_chars = [bsz, seq len, n_chars, emb dim]

        embedded_chars = embedded_chars.unsqueeze(2) #need to add a dummy 'channel' dimension

        print(embedded_chars.shape)

        #embedded_chars = [bsz,seq len, 1, n_chars * emb dim]

        cnn1_output = F.tanh(self.cnn1(embedded_chars).squeeze(-1))
        cnn2_output = F.tanh(self.cnn2(embedded_chars).squeeze(-1))
        cnn3_output = F.tanh(self.cnn3(embedded_chars).squeeze(-1))
        cnn4_output = F.tanh(self.cnn4(embedded_chars).squeeze(-1))
        cnn5_output = F.tanh(self.cnn5(embedded_chars).squeeze(-1))

        print(cnn1_output.shape)

        #each cnn_output = [bsz, seq len, n_filters[i], n_chars-filter_width[i]]        

        cnn1_max = F.max_pool2d(cnn1_output, (1, cnn1_output.shape[3])).squeeze(-1) 
        cnn2_max = F.max_pool2d(cnn2_output, (1, cnn2_output.shape[3])).squeeze(-1) 
        cnn3_max = F.max_pool2d(cnn3_output, (1, cnn3_output.shape[3])).squeeze(-1) 
        cnn4_max = F.max_pool2d(cnn4_output, (1, cnn4_output.shape[3])).squeeze(-1) 
        cnn5_max = F.max_pool2d(cnn5_output, (1, cnn5_output.shape[3])).squeeze(-1) 

        print(cnn1_max.shape)
        
        #each cnn_max = [bsz, seq len, n_filters[i]]

        cnns_max = torch.cat((cnn1_max, cnn2_max, cnn3_max, cnn4_max, cnn5_max), dim=2)

        #cnns_max = [bsz, seq len, sum(n_filters)] 

        print(cnns_max.shape)

        assert 1 == 2

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