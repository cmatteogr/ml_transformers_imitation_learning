import torch
import torch.nn as nn
import math


class sLTM(nn.Module):
    """
    sLTM block, the core of the xSLTM model.
    This implementation is based on the paper: "xLSTM: Extended Long Short-Term Memory"
    https://arxiv.org/abs/2405.04517
    """
    def __init__(self, input_size, hidden_size, proj_size):
        super(sLTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size

        # Input gate
        self.W_i = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.R_i = nn.Parameter(torch.Tensor(hidden_size, proj_size))
        # Forget gate
        self.W_f = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.R_f = nn.Parameter(torch.Tensor(hidden_size, proj_size))
        # Output gate
        self.W_o = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.R_o = nn.Parameter(torch.Tensor(hidden_size, proj_size))
        # Cell gate
        self.W_c = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.R_c = nn.Parameter(torch.Tensor(hidden_size, proj_size))

        # Exponential gating components
        self.W_z = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.R_z = nn.Parameter(torch.Tensor(hidden_size, proj_size))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights for better convergence
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, states):
        h, c, n, m = states

        # Normal LSTM gates
        i_t = torch.sigmoid(x @ self.W_i.t() + h @ self.R_i.t())
        f_t = torch.sigmoid(x @ self.W_f.t() + h @ self.R_f.t())
        o_t = torch.sigmoid(x @ self.W_o.t() + h @ self.R_o.t())
        c_tilde = torch.tanh(x @ self.W_c.t() + h @ self.R_c.t())

        # Exponential Gating
        z_t = torch.sigmoid(x @ self.W_z.t() + h @ self.R_z.t())

        # Update stabilizer states
        m_t = torch.max(f_t * m, i_t)
        n_t = f_t * n + i_t

        # Update cell and hidden states
        c_t = (m_t * (f_t * c * torch.exp(i_t - m_t)) +
               n_t * (i_t * c_tilde * torch.exp(m_t - n_t)))

        h_t = o_t * c_t

        return h_t, (h_t, c_t, n_t, m_t)