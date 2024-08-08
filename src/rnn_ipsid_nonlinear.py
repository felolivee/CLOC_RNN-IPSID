import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

###################################################################################################################################################################

#RNN STAGE 1 - learns behaviroally relevant states by minimizing behavioral loss

#helper class
class Optim1Cell(nn.Module):
    def __init__(self, x_size, y_size, u_size):
        super(Optim1Cell, self).__init__()
        self.xNext_size = x_size
        self.W_A = nn.Linear(x_size, self.xNext_size, dtype= torch.double)
        self.W_K = nn.Linear(y_size, self.xNext_size, dtype= torch.double)
        self.W_B = nn.Linear(u_size, self.xNext_size, dtype= torch.double)

    def forward(self, xBehav_t, y_t, u_t):
        xNext = self.W_A(xBehav_t) + self.W_K(y_t) + self.W_B(u_t)
        return xNext

#helper class
class NonlinearOptim1CellZ(nn.Module):
    def __init__(self, xBehav_size, zhat_size):
        super(NonlinearOptim1CellZ, self).__init__()
        self.Cz1_lin = nn.Linear(xBehav_size, xBehav_size, dtype=torch.double)
        self.Cz1_relu = nn.ReLU()
        self.Cz1_lin2 = nn.Linear(xBehav_size, zhat_size, dtype=torch.double)
        self.Cz1_sigmoid = nn.Sigmoid()
    def forward(self, xBehav_t):
        z_pred = self.Cz1_lin(xBehav_t)
        z_pred = self.Cz1_relu(z_pred)
        z_pred = self.Cz1_lin2(z_pred)
        z_pred = self.Cz1_sigmoid(z_pred)
        return z_pred
    
class Optim1(nn.Module):
    def __init__(self, xBehav_size, y_size, u_size, yhat_size, zhat_size):
        super(Optim1, self).__init__()
        self.xBehav_size = xBehav_size
        self.yhat_size = yhat_size
        self.rnn_cell = Optim1Cell(xBehav_size, y_size, u_size)
        self.nonlinear_cellZ = NonlinearOptim1CellZ(xBehav_size, zhat_size)

    def forward(self, y, u):
        # Assuming y and u are of shape (batch_size, seq_len, feature_size)
        batch_size, seq_len, _ = y.size()

        # Initialize behavioral state with zeros
        xBehav_t = torch.zeros(batch_size, self.xBehav_size, dtype= torch.double).to(y.device)
        
        # Iterate over the sequence
        behavStates = [] #xBehav
        z_hats1 = [] #Cz1(xBehav)

        #initialize nonlinear regressor from behavioral state to neural (optimizer 2)
        for t in range(seq_len):
            behavStates.append(xBehav_t)
            y_t = y[:,t,:]
            u_t = u[:,t, :]
            x_next = self.rnn_cell(xBehav_t, y_t, u_t) #linear forward pass into custom rnn cell
            z_pred = self.nonlinear_cellZ(xBehav_t) #forward pass from x_t+1 (nonlinear combination of x_t+1 -> z_t+1) included in backprop
            z_hats1.append(z_pred)
            xBehav_t = x_next #x_t+1 becomes x_t for next time step (t) in for loop
        
        # Stack the outputs to get a tensor of shape (batch_size, seq_len, output_size)
        behavStates = torch.stack(behavStates, dim=1) #xBehav
        z_hats1 = torch.stack(z_hats1, dim=1) #Cz1(xBehav), used for the loss function and back prop of Stage 1
        return z_hats1, behavStates

class Optim2(nn.Module):
    def __init__(self, xBehav_size, yhat_size):
        super(Optim2, self).__init__()
        self.Cy1_lin = nn.Linear(xBehav_size, xBehav_size, dtype=torch.double)
        self.Cy1_relu = nn.ReLU()
        self.Cy1_lin2 = nn.Linear(xBehav_size, yhat_size, dtype=torch.double)
        self.Cy1_sigmoid = nn.Sigmoid()
    def forward(self, xBehav_t):
        # Assuming xBehav_t has shape (batch_size, seq_len, feature_size)
        _, seq_len, _ = xBehav_t.size()

        y_hats1 = [] #Cy1(xBehav)

        for t in range(seq_len):
            x_t = xBehav_t[:,t,:]
            y_pred = self.Cy1_lin(x_t)
            y_pred = self.Cy1_relu(y_pred)
            y_pred = self.Cy1_lin2(y_pred)
            y_pred = self.Cy1_sigmoid(y_pred)
            y_hats1.append(y_pred)
        
        y_hats1 = torch.stack(y_hats1, dim=1)
        return y_hats1
    
###################################################################################################################################################################

#RNN STAGE 2 - learns any additional dynamics in neural activity that were not learned in the first stage by minimizng neural loss

#helper class
class Optim3Cell(nn.Module):
    def __init__(self, xBehavioral_size, xNeural_size, y_size, u_size):
        super(Optim3Cell, self).__init__()
        self.W_A = nn.Linear(xNeural_size, xNeural_size, dtype=torch.double)
        self.W_K = nn.Linear(y_size + xBehavioral_size, xNeural_size, dtype=torch.double)
        self.W_B = nn.Linear(u_size, xNeural_size, dtype=torch.double)

    def forward(self, xNeural_t, yAndxBehav_t, u_t):
        xNext = self.W_A(xNeural_t) + self.W_K(yAndxBehav_t) + self.W_B(u_t)
        return xNext

#helper class
class NonlinearOptim3CellY(nn.Module):
    def __init__(self, xNeural_size, yhat_size):
        super(NonlinearOptim3CellY, self).__init__()
        self.Cy2_lin = nn.Linear(xNeural_size, xNeural_size, dtype=torch.double)
        self.Cy2_relu = nn.ReLU()
        self.Cy2_lin2 = nn.Linear(xNeural_size, yhat_size, dtype=torch.double)
        self.Cy2_sigmoid = nn.Sigmoid()
    def forward(self, xNeural_t):
        y_pred = self.Cy2_lin(xNeural_t)
        y_pred = self.Cy2_relu(y_pred)
        y_pred = self.Cy2_lin2(y_pred)
        y_pred = self.Cy2_sigmoid(y_pred)
        return y_pred
    
class Optim3(nn.Module):
    def __init__(self,xBehavioral_size, xNeural_size, y_size, u_size, yhat_size, zhat_size):
        super(Optim3, self).__init__()
        self.xNeural_size = xNeural_size
        self.zhat_size = zhat_size
        self.rnn_cell = Optim3Cell(xBehavioral_size, xNeural_size, y_size, u_size)
        self.nonlinear_cellY = NonlinearOptim3CellY(xNeural_size, yhat_size)
    def forward(self, behavStates, y, u):
        # Assuming y and u are of shape (batch_size, seq_len, feature_size)
        batch_size, seq_len, _ = y.size()

        # Initialize neural state with zeros
        xNeural_t = torch.zeros(batch_size, self.xNeural_size, dtype= torch.double).to(y.device)

        #horizontally combine neural data (y) and behavioral latent states (behavStates)
        yAndxBehav = torch.cat((y, behavStates), dim = 2)

        # Iterate over the sequence
        y_hats2 = [] #Cy2(xNeural)
        xNeural_states = [] 

        for t in range(seq_len):
            xNeural_states.append(xNeural_t)
            yAndxBehav_t = yAndxBehav[:,t,:]
            u_t = u[:,t, :]
            x_next = self.rnn_cell(xNeural_t, yAndxBehav_t, u_t) #forward pass into custom rnn cell
            y_pred = self.nonlinear_cellY(xNeural_t) #forward pass from x_t+1 (nonlinear combination of x_t+1 -> y_t+1) included in backprop
            y_hats2.append(y_pred)
            xNeural_t = x_next #x_t+1 becomes x_t for next time step (t) in for loop
        
        # Stack the outputs to get a tensor of shape (batch_size, seq_len, output_size)
        y_hats2 = torch.stack(y_hats2, dim=1)
        xNeural_states = torch.stack(xNeural_states, dim=1)

        return y_hats2, xNeural_states

class Optim4(nn.Module):
    def __init__(self, xNeural_size, zhat_size):
        super(Optim4, self).__init__()
        self.Cz2_lin = nn.Linear(xNeural_size, xNeural_size, dtype=torch.double)
        self.Cz2_relu = nn.ReLU()
        self.Cz2_lin2 = nn.Linear(xNeural_size, zhat_size, dtype=torch.double)
        self.Cz2_sigmoid = nn.Sigmoid()
    def forward(self, xNeural_t):
        # Assuming xBehav_t has shape (batch_size, seq_len, feature_size)
        _, seq_len, _ = xNeural_t.size()

        z_hats2 = [] #Cz2
        for t in range(seq_len):
            x_t = xNeural_t[:,t,:]
            z_pred = self.Cz2_lin(x_t)
            z_pred = self.Cz2_relu(z_pred)
            z_pred = self.Cz2_lin2(z_pred)
            z_pred = self.Cz2_sigmoid(z_pred)
            z_hats2.append(z_pred)

        z_hats2 = torch.stack(z_hats2, dim=1)
        return z_hats2
    
        
