#####################
# import data
#####################
import numpy as np

data=np.loadtxt('./housing.csv')

row, col =data.shape
np.random.shuffle(data) # This function only shuffles the array along the first axis of a multi-dimensional array. 


import torch
import torch.nn as nn

offset=int(row*0.791)
TrainX=data[:offset,:-1]
TrainY=data[:offset,-1]
TestX=data[offset:,:-1]
TestY=data[offset:,-1]

#####################
# bulid model
#####################

Num_X=col-1
n_hiddens = 10
epoch_size=100
batch_size=20

class MLP_regression(nn.Module):
    def __init__(self, input_dims, n_hiddens):
        super(MLP_regression, self).__init__()
        self.input_size = input_dims
        self.hidden_size  = n_hiddens
        
        self.linear_1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(self.hidden_size)
        self.linear_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.8)
        self.linear_3 = nn.Linear(self.hidden_size, 1)
        
    def forward(self, input_tensor):
        hl_1 = self.linear_1(input_tensor)
        hl_1 = self.relu(hl_1)
        hl_1 = self.batchnorm(hl_1)
        hl_2 = self.linear_2(hl_1)
        hl_2 = self.relu(hl_2)
        hl_2 = self.dropout(hl_2)
        output = self.linear_3(hl_2)

        return output



#####################
# compile and fit
#####################

model = MLP_regression(Num_X,n_hiddens)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())



for epoch in range(epoch_size):
    for index, offset in enumerate(range(0, TrainX.shape[0], batch_size)):
        batch_x, batch_y = torch.FloatTensor(TrainX[offset: offset + batch_size]), \
                                        torch.FloatTensor(TrainY[offset: offset + batch_size])
        model.train()
        optimizer.zero_grad()
        train_output = model(batch_x)
        train_loss = criterion(train_output.squeeze(), batch_y)
        
        print(train_loss)
        train_loss.backward()
        optimizer.step()
    


#####################
# evaluate
#####################
test_x=torch.FloatTensor(TestX)
test_y=torch.FloatTensor(TestY)

model.eval()
test_loss = criterion(model(test_x).squeeze(), test_y) 
print('After Training, test loss is ', test_loss.item())

#####################
# predict
#####################
model.eval()
model(test_x)
