import torch
import torch.nn as nn

from src.supervised_baselines.models.sltm import sLTM


class xSLTM(nn.Module):
    """
    xSLTM Model for sequence classification.
    It uses one or more sLTM layers followed by a linear classifier head.
    """
    def __init__(self, input_size, hidden_size, num_layers, n_classes, dropout=0.1):
        super(xSLTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create multiple sLTM layers
        self.sltms = nn.ModuleList([
            sLTM(input_size if i == 0 else hidden_size, hidden_size, hidden_size)
            for i in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize states for each layer
        layer_states = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_size, device=device)
            c = torch.zeros(batch_size, self.hidden_size, device=device)
            n = torch.zeros(batch_size, self.hidden_size, device=device)
            m = torch.zeros(batch_size, self.hidden_size, device=device)
            layer_states.append((h, c, n, m))

        # Process sequence step-by-step
        for t in range(seq_len):
            input_t = x[:, t, :]
            for i in range(self.num_layers):
                h_next, new_states = self.sltms[i](input_t, layer_states[i])
                layer_states[i] = new_states
                input_t = self.dropout(h_next) # Apply dropout between layers

        # Use the final hidden state for classification
        last_hidden_state = layer_states[-1][0]
        output = self.classifier(last_hidden_state)
        return output