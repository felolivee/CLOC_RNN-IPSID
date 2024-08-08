import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#helper class
class CustomRNNCell(nn.Module):
    def __init__(self, x_size, y_size, u_size):
        super(CustomRNNCell, self).__init__()
        self.xNext_size = x_size
        self.W_A = nn.Linear(x_size, self.xNext_size, dtype=torch.double)
        self.W_K = nn.Linear(y_size, self.xNext_size, dtype=torch.double)
        self.W_B = nn.Linear(u_size, self.xNext_size, dtype=torch.double)

    def forward(self, x_t, y_t, u_t):
        print(x_t.dtype)
        print(y_t.dtype)
        print(u_t.dtype)
        xNext = self.W_A(x_t) + self.W_K(y_t) + self.W_B(u_t)
        return xNext

class CustomRNN(nn.Module):
    def __init__(self, x_size, y_size, u_size, yhat_size, zhat_size):
        super(CustomRNN, self).__init__()
        self.x_size = x_size
        self.rnn_cell = CustomRNNCell(x_size, y_size, u_size)
        self.W_Cy = nn.Linear(x_size, yhat_size, dtype=torch.double)
        self.W_Cz = nn.Linear(x_size, zhat_size, dtype=torch.double)

    def forward(self, y, u):
        # Assuming y and u are of shape (batch_size, seq_len, feature_size)
        batch_size, seq_len, _ = y.size()

        # Initialize the hidden state
        x_t = torch.zeros(batch_size, self.x_size, dtype= torch.double).to(y.device)
        
        # Iterate over the sequence
        y_hats = []
        z_hats = []
        for t in range(seq_len):
            y_t = y[:,t,:]
            u_t = u[:,t, :]
            x_next = self.rnn_cell(x_t, y_t, u_t) #forward pass into custom rnn cell
            y_pred = self.W_Cy(x_t) #forward pass from x_t+1 (linear combination of x_t+1 -> y_t+1)
            z_pred = self.W_Cz(x_t) #forward pass from x_t+1 (linear combination of x_t+1 -> z_t+1)
            y_hats.append(y_pred)
            z_hats.append(z_pred)
            x_t = x_next #x_t+1 becomes x_t for next time step (t) in for loop
        
        # Stack the outputs to get a tensor of shape (batch_size, seq_len, output_size)
        y_hats = torch.stack(y_hats, dim=1)
        z_hats = torch.stack(z_hats, dim=1)
        return y_hats, z_hats

        


