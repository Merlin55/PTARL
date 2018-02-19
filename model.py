#-------------------------------------------------------------------------------
# Name:        model implementation 
# Purpose:     for the kdd'18 sptia-temporal representation learning
#
# Author:      Pengyang Wang
#
# Created:     1/23/2018
# Copyright:   (c) Pengyang Wang@2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import os 
import pandas as pd
import pickle



# 
class AEGRU(torch.nn.Module):
    """
    Implementation of the framework with auto-encoder 
    integrated with GRU (Gated Recurrent Unit)

    """

    def __init__(self, n_feature, n_hidden1, n_hidden2, hidden_size, n_output):
        '''
        initialization of the class, construct the network structure
        '''
        super(AEGRU, self).__init__()
        self.encoder_l1 = torch.nn.Linear(n_feature, 
                                          n_hidden1)  # first layer of the encoder
#         self.encoder_l2 = torch.nn.Linear(n_hidden1, 
#                                           hidden_size)
        self.gru = torch.nn.GRU(          # middle layer of the frame GRU
            input_size=n_hidden1, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True)     
        self.decoder_l2 = torch.nn.Linear(hidden_size,  # the last 2 layer of the decoder
                                          n_hidden1)
        self.decoder_l1 = torch.nn.Linear(n_hidden1,    # the last 1 layer of the decoder
                                          n_output)
    
      
    def forward(self, x, encoded):
        '''
        forward function, define the computation process of the network 
        '''
        x = F.sigmoid(self.encoder_l1(x))           # use sigmoid function as the motivation function to the encoder layer
#         x2 = F.sigmoid(self.encoder_l2(x))
#         encoded = F.sigmoid(self.encoder_l2(x))
        encoded_all, encoded = self.gru(x, encoded) # for GRU in the middle encoded vector as the hidden state, output 
                                                    # of the encoder as the input, return two values, first is the all the
                                                    # hidden states (encoded vectors), second is the hidden state (encoded, vector)
                                                    # at the current time  
        x = F.sigmoid(self.decoder_l2(encoded_all)) # use sigmoid function as the motivation function to the encoder layer
        decoded = F.sigmoid(self.decoder_l1(x))
        return encoded_all, encoded, decoded
    

    
    


class Model_Loss(torch.nn.Module):
    """
    Implementation of the customized Loss Function
    """
    def __init__(self):
        super(Model_Loss, self).__init__()
        
    def forward(self, mseloss, decoded, y, sim_tensor, encoded_all, ALPHA):
        '''
        mseloss: the instance of torch.nn.MSELoss(), for MSE loss
        decoded: decoded results of the framework, format-(batch_size, time, original_feature_size)
        y: ground truth, original features, format-(batch_size, time, original_feature_size)
        sim_tensor: similarity tensor for z_i, z_j at time t, format-(batch_size, time, batch_size)
        encoded_all: encoded vectors for all time t, format-(batch_size, time, encoded_feature_size)
        '''     

        loss1 = mseloss(decoded, y)                                   # first part of the loss function, mse of the x and \hat{x}
        z_i_j_t = encoded_all.unsqueeze(1) -  encoded_all.unsqueeze(0)    # second part of the loss function
                                                                          # pairwise subtraction between z_i and z_j at time t
        z_i_j_t_square = (z_i_j_t**2).sum(dim=3)                      # square of the subtraction
        loss2 = (((sim_tensor * z_i_j_t_square).sum(dim=2)).sum(dim=1)).sum(dim=0) # summation of all pairwise square of the subtraction
        loss = loss1 + ALPHA*loss2   
        return loss
        

def getIndex(data):
    '''
    keep index for shuffled dataset
    '''
    data_size = data[:, :, -1].size()
    origin_index = data[:, :, -1].numpy()
    index = list(origin_index[:, 0])
    return map(int, index)
    
def save_result(data, ALPHA, EPOCH, LR, view):
    path = '/home/pengyang/Documents/code/KDD18-spatiotemporal/model_result/' + view
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = '%s/result_alpha_%s_epoch_%s_lr_%s.pkl' % (path, ALPHA, EPOCH, LR)
    f = open(save_path, 'wb')
    pickle.dump(data, f)
    f.close()
    print 'result_alpha_%s_epoch_%s_lr_%s saved' % (ALPHA, EPOCH, LR)

def prepare_data(apath, view):
    f1 = open(os.path.join(apath, 'data_loader.pkl'), 'rb')
    loader = pickle.load(f1) 
    f1.close()
    f2 = open(os.path.join(apath, 'similarity_%s.pkl' % view), 'rb')
    simlarity_matrix = pickle.load(f2)
    f2.close()
    return loader, simlarity_matrix

def prepare_similarity(apath, view='prob'):
    path = os.path.join(apath, 'view_'+view)
    sim = []
    for i in range(1, 21):
            file_path = path + '/' + str(i)
            cur_sim = pd.read_csv(file_path, header=None).as_matrix()
            sim.append(cur_sim)
    sim = np.array(sim)
    sim = np.moveaxis(sim, 0, 1)
    return sim      
        
# Hyperparameters        
#ALPHA = 0.1  # controlling spatial regularizer
#EPOCH = 3    # train times
#LR = 0.0002  # learning rate
N_FEATURE = N_OUTPUT = 81  # original and decoded feature size
N_HIDDEN1 = 40             # size of the first group layer for auto-encoder
N_HIDDEN2 = 10              # size of the second group layer for auto-encoder
HIDDEN_SIZE = 20            # size of the encoded layer/hidden state/model output
BATCH_SIZE = 301


# initializations
# aegru = AEGRU(N_FEATURE, N_HIDDEN1, N_HIDDEN2, HIDDEN_SIZE, N_OUTPUT)  # initialization of the model 
# optimizer = torch.optim.SGD(aegru.parameters(), lr=LR)         # initialization of the optimizer, using SGD
# mseloss = torch.nn.MSELoss()   # initialization of the MSE loss function
# loss_func = Model_Loss()       # initialization of the customized loss function for the model

# data preparation
view = 'time'
path_data = '/home/Data/Pengyang/T-drive-taxi/transformed_dataset'
path_similarity = '/home/Data/Pengyang/T-drive-taxi/dataset_similarity'
ppath = '/home/Data/Pengyang/T-drive-taxi/train_dataset'
loader, sim_mat = prepare_data(ppath, view)
sim_tensor = torch.Tensor(sim_mat)
sim_tensor = sim_tensor.permute(1, 0, 2)
# x = torch.Tensor(prepare_data(path_data, view))  # input data
# index = [np.ones((20, 1))*i for i in range(int(x.size()[0]))] # generate indexes to remember the location of the shuffled input data
# index = torch.Tensor(index)
# sim_tensor = torch.Tensor(prepare_similarity(path_similarity, view))  # similarity sensor
# data = torch.cat((x, index), dim=2)    # concat indexes with input data to utilize the index
# torch_dataset = Data.TensorDataset(data_tensor=data, target_tensor=data) # transform tensor to tensor dataset
# loader = Data.DataLoader(             # transform tensor dataset to data loader
#     dataset=torch_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=2, #Thread to use
# )



for ALPHA in (10, 1, 0.5, 0.2, 0.1, 0.05, 0.01):
    for EPOCH in range(5, 50, 5):
        for LR in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
            print '====alpha:%s====epoch:%s====lr:%s===' % (ALPHA, EPOCH, LR)
            # encoded = None  # initialization of the hidden state
            
            aegru = AEGRU(N_FEATURE, N_HIDDEN1, N_HIDDEN2, HIDDEN_SIZE, N_OUTPUT)
            optimizer = torch.optim.SGD(aegru.parameters(), lr=LR)         # initialization of the optimizer, using SGD
            mseloss = torch.nn.MSELoss()   # initialization of the MSE loss function
            loss_func = Model_Loss()       # initialization of the customized loss function for the model
            for epoch in range(EPOCH):    # training for each epoch
                hidden_states = []
                # indexes = []
                for step, (data, target) in enumerate(loader):  # training for each batch
                    encoded = None
            #         if epoch == 0 and step == 0:
                    if view == 'prob':
                        x = data[:, :, :81].contiguous()   # re-construct input data
                    else:
                        print view
                        x = data[:, :, 82:-1].contiguous()
                    cur_index = getIndex(data)         # get current index for the shuffled data
                    # indexes = indexes + cur_index
                    sub_sim_tensor = sim_tensor[cur_index, :, :]     # get the according similarity tensor based on the index
                    sub_sim_tensor = sub_sim_tensor[:, :, cur_index] 
                    sub_sim_tensor = sub_sim_tensor.permute(0, 2, 1)
                    b_x = Variable(x.view(-1, 20, N_FEATURE))       # transform  tensors into Variable
                    b_y = Variable(x.view(-1, 20, N_FEATURE))
                    sub_sim_tensor = Variable(sub_sim_tensor)
                    encoded_all, encoded, decoded = aegru(b_x, encoded)
                    loss = loss_func(mseloss, decoded, b_y, sub_sim_tensor, encoded_all, ALPHA)
                    optimizer.zero_grad()               # clear gradients for this training step
                    loss.backward(retain_graph=True)    # back-propagation, compute gradients
                    optimizer.step()                    # apply gradients  
                    # if step % 2 == 0:
                    print 'epoch: %d| step: %d| loss: %f' % (epoch, step, loss.data)
                    hidden_states.append(encoded_all)
            # index_hiden_states = dict(zip(indexes, hidden_states))    
            save_path = '/home/pengyang/Documents/code/KDD18-spatiotemporal/model_parameter/view_time'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            f_path = '%s/alpha_%s_epoch_%s_lr_%s.pkl' % (save_path, ALPHA, EPOCH, LR)
            torch.save(aegru.state_dict(), f_path)
            resultt = hidden_states[0]
            for i in range(1, len(hidden_states)):
                resultt = torch.cat((resultt, hidden_states[i]), dim=0)
            save_result(resultt, ALPHA, EPOCH, LR, view)

            
        








