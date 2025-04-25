import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import data_set


def accuracy_fn(y_true, y_pred):
  correct =torch.eq(y_true, y_pred).sum().item()
  acc=(correct/len(y_pred))*100
  return acc


# extraer los datos
X=data_set.df_songs['Views'].values
y=data_set.df_songs['Stream'].values
# transformar los datos en tensors
X=torch.from_numpy(np.array(X)).type(torch.float).view(-1, 1)
y=torch.from_numpy(np.array(y)).type(torch.float)
# variable device para usar gpu en caso de que se pueda
device= "cuda" if torch.cuda.is_available() else "cpu"

class LinearRegressionModel(nn.Module): # almost everything in pytorch inheritance from nn
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(in_features=1, out_features=100)
    self.layer_2 = nn.Linear(in_features=100, out_features=100)
    self.layer_3 = nn.Linear(in_features=100, out_features=1)
    # foward() defines the computation in the model
  def forward(self,x:torch.Tensor)->torch.Tensor: # x is the input data
    return self.layer_3(torch.relu(self.layer_2(torch.relu(self.layer_1(x)))))

# se crea un modelo de regresion linear
linear_regression_model=LinearRegressionModel().to(device)

# semilla para que el modelo sea replicable
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# función de pérdida (Sigmoid + Binary cross entropy)
loss_fn=nn.L1Loss()

# optimizador
optimizer=torch.optim.SGD(params=linear_regression_model.parameters(), lr=0.01)
# partición de los datos
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

# enviar los datos al device
X_train,y_train=X_train.to(device),y_train.to(device)
X_test,y_test=X_test.to(device),y_test.to(device)
# training loop

epochs=10000
for epoch in range(epochs):
  # training
  linear_regression_model.train()
  # foward pass
  y_pred=linear_regression_model(X_train).squeeze()

  # calcular loss y acc
  loss=loss_fn(y_pred, y_train)
  acc=accuracy_fn(y_true=y_train,y_pred=y_pred)

  # optimizer
  optimizer.zero_grad()

  # loss backward
  loss.backward()

  # optimizer step
  optimizer.step()

  ### testing
  linear_regression_model.eval()
  with torch.inference_mode():
    # forward pass
    test_pred=linear_regression_model(X_test).squeeze()
    #  test loss
    test_loss=loss_fn(test_pred, y_test)
    test_acc=accuracy_fn(y_true=y_test,y_pred=test_pred)

  #print
  if epoch %10==0:
    print(f"Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.2f}% | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}%")
