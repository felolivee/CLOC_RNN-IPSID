import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import torch
import pandas as pd


class GordonStanleyDataProcessor():
    def __init__(self, raw_filepath, thetaco_filepath, batch_size, sequence_legth, hasBehavioral, training_size = 0.8, remove_last_batch = True):
        self.raw_filepath = raw_filepath
        self.thetaco_filepath = thetaco_filepath
        self.batch_size = batch_size
        self.sequence_length = sequence_legth
        self.hasBehav = hasBehavioral
        self.training_size = training_size
        self.remove_last_batch = remove_last_batch
    
    def processGordon(self):
        y, target_y, u, z, target_z, time_points = self.readGordon()
        y_train, y_test, target_y_train, target_y_test, u_train, u_test, z_train, z_test, target_z_train, target_z_test = self.splitGordon(y, target_y, u, z, target_z)
        y_train_tensor, y_test_tensor, target_y_train_tensor, target_y_test_tensor, u_train_tensor, u_test_tensor, z_train_tensor, z_test_tensor, target_z_train_tensor, target_z_test_tensor = self.tensorGordon(y_train, y_test, target_y_train, target_y_test, u_train, u_test, z_train, z_test, target_z_train, target_z_test)
        y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, z_train_batched, z_test_batched, target_z_train_batched, target_z_test_batched = self.batchGordon(y_train_tensor, y_test_tensor, target_y_train_tensor, target_y_test_tensor, u_train_tensor, u_test_tensor, z_train_tensor, z_test_tensor, target_z_train_tensor, target_z_test_tensor)
        if (self.remove_last_batch):
            return self.stackRemoveGordon(y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, z_train_batched, z_test_batched, target_z_train_batched, target_z_test_batched)
        else:
            return self.stackGordon(y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, z_train_batched, z_test_batched, target_z_train_batched, target_z_test_batched)

    #helper fxn
    def readGordon(self):
        theta_and_behavior = pd.read_csv(self.thetaco_filepath, delim_whitespace=True)
        if (self.raw_filepath == ''):
            #get y
            theta_co = theta_and_behavior['Theta_coherence']
            y = theta_co
            y = np.reshape(y, (y.shape[0], 1))
            target_y = y[1:]
            last_rowy = y[-1]
            target_y = np.vstack([target_y, last_rowy])
        else:
            raw_lfp = pd.read_csv(self.raw_filepath, delim_whitespace=True)
            #get y
            vHPC_LFP = raw_lfp['vHPC_LFP']
            vHPC_LFP = np.reshape(vHPC_LFP, (vHPC_LFP.shape[0], 1))
            mPFC_LFP = raw_lfp['mPFC_LFP']
            mPFC_LFP = np.reshape(mPFC_LFP, (mPFC_LFP.shape[0], 1))
            y = np.concatenate((vHPC_LFP, mPFC_LFP), axis = 1)
            target_y = y[1:]
            last_rowy = y[-1]
            target_y = np.vstack([target_y, last_rowy])


        #get u
        show = theta_and_behavior['Show']
        show = np.reshape(show, (show.shape[0],1))
        delay = theta_and_behavior['Delay']
        delay = np.reshape(delay, (delay.shape[0],1))
        test = theta_and_behavior['Test']
        test = np.reshape(test, (test.shape[0],1))
        rest = theta_and_behavior['Rest']
        rest = np.reshape(rest, (rest.shape[0],1))
        control_input = theta_and_behavior['Control_input']
        control_input = np.reshape(control_input, (control_input.shape[0],1))
        u = np.hstack([show, delay, test, rest, control_input])

        #get z
        correct = theta_and_behavior['Correct']
        z = correct
        z = np.reshape(z, (z.shape[0],1))
        target_z = z[1:]
        last_rowz = z[-1]
        target_z = np.vstack([target_z, last_rowz])
        
        time_points = theta_and_behavior['Time_points']
        time_points = np.reshape(time_points, (time_points.shape[0],1))
        return y, target_y, u, z, target_z, time_points
    
    #helper fxn
    def splitGordon(self, y, target_y, u, z, target_z):
        #split into training and testing
        train_size = int(self.training_size * y.shape[0])

        y_train, y_test = y[:train_size], y[train_size:]
        target_y_train, target_y_test = target_y[:train_size], target_y[train_size:]
        u_train, u_test = u[:train_size], u[train_size:]
        z_train, z_test = z[:train_size], z[train_size:]
        target_z_train, target_z_test = target_z[:train_size], target_z[train_size:]

        return y_train, y_test, target_y_train, target_y_test, u_train, u_test, z_train, z_test, target_z_train, target_z_test

    #helper fxn
    def tensorGordon(self, y_train, y_test, target_y_train, target_y_test, u_train, u_test, z_train, z_test, target_z_train, target_z_test):    
        #convert data into tensors
        y_train_tensor = torch.from_numpy(y_train)
        y_test_tensor = torch.from_numpy(y_test)

        target_y_train_tensor = torch.from_numpy(target_y_train)
        target_y_test_tensor = torch.from_numpy(target_y_test)

        u_train_tensor = torch.from_numpy(u_train)
        u_test_tensor = torch.from_numpy(u_test)

        z_train_tensor = torch.from_numpy(z_train)
        z_test_tensor = torch.from_numpy(z_test)

        target_z_train_tensor = torch.from_numpy(target_z_train)
        target_z_test_tensor = torch.from_numpy(target_z_test)


        return y_train_tensor, y_test_tensor, target_y_train_tensor, target_y_test_tensor, u_train_tensor, u_test_tensor, z_train_tensor, z_test_tensor, target_z_train_tensor, target_z_test_tensor
    
    #helper fxn
    def batchGordon(self, y_train_tensor, y_test_tensor, target_y_train_tensor, target_y_test_tensor, u_train_tensor, u_test_tensor, z_train_tensor, z_test_tensor, target_z_train_tensor, target_z_test_tensor):
        #get tensors into batches of desired sequence length
        y_train_batched = torch.split(y_train_tensor, self.sequence_length)
        y_test_batched = torch.split(y_test_tensor, self.sequence_length)

        target_y_train_batched = torch.split(target_y_train_tensor, self.sequence_length)
        target_y_test_batched = torch.split(target_y_test_tensor, self.sequence_length)

        u_train_batched = torch.split(u_train_tensor, self.sequence_length)
        u_test_batched = torch.split(u_test_tensor, self.sequence_length)

        z_train_batched = torch.split(z_train_tensor, self.sequence_length)
        z_test_batched = torch.split(z_test_tensor, self.sequence_length)

        target_z_train_batched = torch.split(target_z_train_tensor, self.sequence_length)
        target_z_test_batched = torch.split(target_z_test_tensor, self.sequence_length)

        return y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, z_train_batched, z_test_batched, target_z_train_batched, target_z_test_batched

    #helper fxn
    def stackGordon(self, y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, z_train_batched, z_test_batched, target_z_train_batched, target_z_test_batched):
        #stack tuples
        y_train_batched = torch.stack(y_train_batched[:])
        y_test_batched = torch.stack(y_test_batched[:])

        target_y_train_batched = torch.stack(target_y_train_batched[:])
        target_y_test_batched = torch.stack(target_y_test_batched[:])

        u_train_batched = torch.stack(u_train_batched[:])
        u_test_batched = torch.stack(u_test_batched[:])

        z_train_batched = torch.stack(z_train_batched[:])
        z_test_batched = torch.stack(z_test_batched[:])

        target_z_train_batched = torch.stack(target_z_train_batched[:])
        target_z_test_batched = torch.stack(target_z_test_batched[:])

        return y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, z_train_batched, z_test_batched, target_z_train_batched, target_z_test_batched

    #helper fxn
    def stackRemoveGordon(self, y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, z_train_batched, z_test_batched, target_z_train_batched, target_z_test_batched):
        #stack tuples and remove last batch for training data
        y_train_batched = torch.stack(y_train_batched[:-1])
        y_test_batched = torch.stack(y_test_batched[:-1])

        target_y_train_batched = torch.stack(target_y_train_batched[:-1])
        target_y_test_batched = torch.stack(target_y_test_batched[:-1])

        u_train_batched = torch.stack(u_train_batched[:-1])
        u_test_batched = torch.stack(u_test_batched[:-1])

        z_train_batched = torch.stack(z_train_batched[:-1])
        z_test_batched = torch.stack(z_test_batched[:-1])

        target_z_train_batched = torch.stack(target_z_train_batched[:-1])
        target_z_test_batched = torch.stack(target_z_test_batched[:-1])

        return y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, z_train_batched, z_test_batched, target_z_train_batched, target_z_test_batched
    
    def processNWB(self):
        y, target_y, u, z, target_z = self.readNWB()
        print("y shape: {}, target_y shape: {}, u shape: {}".format( y.shape, target_y.shape, u.shape))
        print ("z shape: {}, target_z shape: {}".format(z.shape, target_z.shape))
        y_train, y_test, target_y_train, target_y_test, u_train, u_test, z_train, z_test, target_z_train, target_z_test = self.splitNWB(y, target_y, u, z, target_z)
        print("normal - z train shape: {}, z test shape: {}, target z train shape: {}, target z test shape {}".format(z_train.shape, z_test.shape, target_z_train.shape, target_z_test.shape))
        # u_train, u_test = self.reshape(u_train, u_test)
        y_train_tensor, y_test_tensor, target_y_train_tensor, target_y_test_tensor, u_train_tensor, u_test_tensor, z_train_tensor, z_test_tensor, target_z_train_tensor, target_z_test_tensor = self.tensorNWB(y_train, y_test, target_y_train, target_y_test, u_train, u_test, z_train, z_test, target_z_train, target_z_test)
        print("tesnor - z train shape: {}, z test shape: {}, target z train shape: {}, target z test shape {}".format(z_train_tensor.shape, z_test_tensor.shape, target_z_train_tensor.shape, target_z_test_tensor.shape))
        y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, z_train_batched, z_test_batched, target_z_train_batched, target_z_test_batched = self.batchNWB(y_train_tensor, y_test_tensor, target_y_train_tensor, target_y_test_tensor, u_train_tensor, u_test_tensor, z_train_tensor, z_test_tensor, target_z_train_tensor, target_z_test_tensor)
        print("batched - z train shape: {}, z test shape: {}, target z train shape: {}, target z test shape {}".format(z_train_batched.shape, z_test_batched.shape, target_z_train_batched.shape, target_z_test_batched.shape))
        if (self.remove_last_batch):
            return self.stackRemoveNWB(y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, z_train_batched, z_test_batched, target_z_train_batched, target_z_test_batched)
        else:
            return self.stackNWB(y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, z_train_batched, z_test_batched, target_z_train_batched, target_z_test_batched)

    #helper fxn
    def readNWB(self):
        #get data
        filepath = self.raw_filepath
        # Open the file in read mode "r",
        io = NWBHDF5IO(filepath, mode="r")
        nwbfile = io.read()
        y = nwbfile.processing['ecephys'].data_interfaces['LFP1'].electrical_series['ElectricalSeries'].data[:] * float(1000000) # convert to microvolts 
        target_y = nwbfile.processing['ecephys'].data_interfaces['LFP1'].electrical_series['ElectricalSeries'].data[1:,:] * float(1000000) # convert to microvolts 
        last_row_y = y[-1, :]
        target_y = np.vstack([target_y, last_row_y])
        u_opto = nwbfile.acquisition['OptogeneticSeries1'].data[:]
        u_opto = np.reshape(u_opto, (u_opto.shape[0], 1))
        try:
            u_galvo = nwbfile.acquisition['GalvoSeries1'].data[:]
            u_galvo = np.reshape(u_galvo, (u_galvo.shape[0], 1))
            u = np.concatenate((u_opto, u_galvo), axis = 1)
        except:
            print("this dataset does not have galvo drive component")
            u = u_opto
        if (self.hasBehav):
            z_whisking = nwbfile.processing['whisker'].data_interfaces['Whisking'].data[:]
            z_whisking = np.reshape(z_whisking, (z_whisking.shape[0], 1))
            z_whiskerMotion = nwbfile.processing['whisker'].data_interfaces['WhiskerMotion'].data[:]
            z_whiskerMotion = np.reshape(z_whiskerMotion, (z_whiskerMotion.shape[0], 1))
            z = np.concatenate((z_whisking, z_whiskerMotion), axis = 1)
            last_row_z = z[-1, :]
            target_z = z[1:]
            target_z = np.vstack([target_z, last_row_z])
            return y, target_y, u, z, target_z
        return y, target_y, u, None, None
    
    #helper fxn
    def splitNWB(self, y, target_y, u, z, target_z):
        #split into training and testing
        train_size = int(self.training_size * y.shape[0])

        y_train, y_test = y[:train_size], y[train_size:]
        target_y_train, target_y_test = target_y[:train_size], target_y[train_size:]
        u_train, u_test = u[:train_size], u[train_size:]
        if (self.hasBehav):
            z_train, z_test = z[:train_size], z[train_size:]
            target_z_train, target_z_test = target_z[:train_size], target_z[train_size:]
            return y_train, y_test, target_y_train, target_y_test, u_train, u_test, z_train, z_test, target_z_train, target_z_test
        
        return y_train, y_test, target_y_train, target_y_test, u_train, u_test, None, None, None, None
    
    # #helper fxn
    # def reshape(self, u_train, u_test):
    #     #reshape u to be of size (batch_size, sequence_length, dimensions)
    #     u_train = np.reshape(u_train, (u_train.shape[0], 1))
    #     u_test = np.reshape(u_test, (u_test.shape[0], 1))

    #     return u_train, u_test

    #helper fxn
    def tensorNWB(self, y_train, y_test, target_y_train, target_y_test, u_train, u_test, z_train, z_test, target_z_train, target_z_test):    
        #convert data into tensors
        y_train_tensor = torch.from_numpy(y_train).double()
        y_test_tensor = torch.from_numpy(y_test).double()

        target_y_train_tensor = torch.from_numpy(target_y_train).double()
        target_y_test_tensor = torch.from_numpy(target_y_test).double()

        u_train_tensor = torch.from_numpy(u_train).double()
        u_test_tensor = torch.from_numpy(u_test).double()

        if (self.hasBehav):
            z_train_tensor = torch.from_numpy(z_train).double()
            z_test_tensor = torch.from_numpy(z_test).double()

            target_z_train_tensor = torch.from_numpy(target_z_train).double()
            target_z_test_tensor = torch.from_numpy(target_z_test).double()
            return y_train_tensor, y_test_tensor, target_y_train_tensor, target_y_test_tensor, u_train_tensor, u_test_tensor, z_train_tensor, z_test_tensor, target_z_train_tensor, target_z_test_tensor

        return y_train_tensor, y_test_tensor, target_y_train_tensor, target_y_test_tensor, u_train_tensor, u_test_tensor, None, None, None, None

    #helper fxn
    def batchNWB(self, y_train_tensor, y_test_tensor, target_y_train_tensor, target_y_test_tensor, u_train_tensor, u_test_tensor, z_train_tensor, z_test_tensor, target_z_train_tensor, target_z_test_tensor):
        #get tensors into batches of desired sequence length
        y_train_batched = torch.split(y_train_tensor, self.sequence_length)
        y_test_batched = torch.split(y_test_tensor, self.sequence_length)

        target_y_train_batched = torch.split(target_y_train_tensor, self.sequence_length)
        target_y_test_batched = torch.split(target_y_test_tensor, self.sequence_length)

        u_train_batched = torch.split(u_train_tensor, self.sequence_length)
        u_test_batched = torch.split(u_test_tensor, self.sequence_length)

        if (self.hasBehav):
            z_train_batched = torch.split(z_train_tensor, self.sequence_length)
            z_test_batched = torch.split(z_test_tensor, self.sequence_length)

            target_z_train_batched = torch.split(target_z_train_tensor, self.sequence_length)
            target_z_test_batched = torch.split(target_z_test_tensor, self.sequence_length)
            return y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, z_train_batched, z_test_batched, target_z_train_batched, target_z_test_batched
        
        return y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, None, None, None, None


    #helper fxn
    def stackNWB(self, y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, z_train_batched, z_test_batched, target_z_train_batched, target_z_test_batched):
        #stack tuples
        y_train_batched = torch.stack(y_train_batched[:])
        y_test_batched = torch.stack(y_test_batched[:])

        target_y_train_batched = torch.stack(target_y_train_batched[:])
        target_y_test_batched = torch.stack(target_y_test_batched[:])

        u_train_batched = torch.stack(u_train_batched[:])
        u_test_batched = torch.stack(u_test_batched[:])

        if (self.hasBehav):
            z_train_batched = torch.stack(z_train_batched[:])
            z_test_batched = torch.stack(z_test_batched[:])

            target_z_train_batched = torch.stack(target_z_train_batched[:])
            target_z_test_batched = torch.stack(target_z_test_batched[:])
            return y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, z_train_batched, z_test_batched, target_z_train_batched, target_z_test_batched
        
        return y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, None, None, None, None


    #helper fxn
    def stackRemoveNWB(self, y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, z_train_batched, z_test_batched, target_z_train_batched, target_z_test_batched):
        #stack tuples and remove last batch for training data
        y_train_batched = torch.stack(y_train_batched[:-1])
        y_test_batched = torch.stack(y_test_batched[:-1])

        target_y_train_batched = torch.stack(target_y_train_batched[:-1])
        target_y_test_batched = torch.stack(target_y_test_batched[:-1])

        u_train_batched = torch.stack(u_train_batched[:-1])
        u_test_batched = torch.stack(u_test_batched[:-1])

        if (self.hasBehav):
            z_train_batched = torch.stack(z_train_batched[:-1])
            z_test_batched = torch.stack(z_test_batched[:-1])

            target_z_train_batched = torch.stack(target_z_train_batched[:-1])
            target_z_test_batched = torch.stack(target_z_test_batched[:-1])

            return y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, z_train_batched, z_test_batched, target_z_train_batched, target_z_test_batched
        
        return y_train_batched, y_test_batched, target_y_train_batched, target_y_test_batched, u_train_batched, u_test_batched, None, None, None, None
