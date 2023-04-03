import numpy as np
import pandas as pd
from sklearn import model_selection
import torch
from FNN import FNN
from scipy.stats import norm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Duality\\Documents\\Sophia+Zhu\\Workplace\\Psychology\\finalStudentInfo.csv")
df.drop(df.columns[0],inplace=True, axis=1)

# df.cov().to_csv('covariance.csv')

end = 0
for index, row in df.iterrows():
    if row['imd_band_0-10%'].item() == 1:
        end += 5
    elif row['imd_band_10-20'].item() == 1:
        end += 15
    elif row['imd_band_20-30%'].item() == 1:
        end += 25
    elif row['imd_band_30-40%'].item() == 1:
        end += 35
    elif row['imd_band_40-50%'].item() == 1:
        end += 45
    elif row['imd_band_50-60%'].item() == 1:
        end += 55
    elif row['imd_band_60-70%'].item() == 1:
        end += 65
    elif row['imd_band_70-80%'].item() == 1:
        end += 75
    elif row['imd_band_80-90%'].item() == 1:
        end += 85
    elif row['imd_band_90-100%'].item() == 1:
        end += 95
end /= len(df)
avg = end
print(avg)
result = 0
for index, row in df.iterrows():
    if row['imd_band_0-10%'].item() == 1:
        end = 5
    elif row['imd_band_10-20'].item() == 1:
        end = 15
    elif row['imd_band_20-30%'].item() == 1:
        end = 25
    elif row['imd_band_30-40%'].item() == 1:
        end = 35
    elif row['imd_band_40-50%'].item() == 1:
        end = 45
    elif row['imd_band_50-60%'].item() == 1:
        end = 55
    elif row['imd_band_60-70%'].item() == 1:
        end = 65
    elif row['imd_band_70-80%'].item() == 1:
        end = 75
    elif row['imd_band_80-90%'].item() == 1:
        end = 85
    elif row['imd_band_90-100%'].item() == 1:
        end = 95
    result += (end-avg)**2
result /= len(df)
result = result**0.5
print(result)

x_train, x_test, y_train, y_test = model_selection.train_test_split(df, df['score'], test_size = 0.2, random_state = 1234)

for i in df.columns:
    if i != 'score':
        print('corr of '+i+': '+str(df['score'].corr(df[i])))

class VLEDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[[idx], :]
        end = -1
        if row['imd_band_0-10%'].item() == 1:
            end = 1
        elif row['imd_band_10-20'].item() == 1:
            end = 2
        elif row['imd_band_20-30%'].item() == 1:
            end = 3
        elif row['imd_band_30-40%'].item() == 1:
            end = 4
        elif row['imd_band_40-50%'].item() == 1:
            end = 5
        elif row['imd_band_50-60%'].item() == 1:
            end = 6
        elif row['imd_band_60-70%'].item() == 1:
            end = 7
        elif row['imd_band_70-80%'].item() == 1:
            end = 8
        elif row['imd_band_80-90%'].item() == 1:
            end = 9
        elif row['imd_band_90-100%'].item() == 1:
            end = 10
        z = (end*10-49.52)/28.17
        band_score = norm.cdf(z)
        edu = -1
        if row['highest_education_No Formal quals'].item() == 1:
            edu = 1
        if row['highest_education_Lower Than A Level'].item() == 1:
            edu = 2
        elif row['highest_education_A Level or Equivalent'].item() == 1:
            edu = 3
        elif row['highest_education_HE Qualification'].item() == 1:
            edu = 4
        elif row['highest_education_Post Graduate Qualification'].item() == 1:
            edu = 5
        age = -1
        if row['age_band_0-35'].item()==1:
            age = 17
        elif row['age_band_35-55'].item()==1:
            age = 45
        else:
            age = 60
        # ex_env = alpha*(row['assessment_type_CMA'].item()*1 + row['assessment_type_TMA'].item()*(-1)) + np.exp(end/10)
        # ex_phy = np.exp(3)*row['age_band_0-35'].item() +  np.exp(2)*row['age_band_35-55'].item() + np.exp(1)*row['age_band_55<='].item()
        in_ach = beta*row['score'].item()**2 - (row['interact_times'].item()**2) - (row['num_of_prev_attempts'].item()**2)
        in_aff = (np.log(row['score'].item()+1) + band_score + alpha*row['num_of_prev_attempts'].item() + alpha*edu)/alpha
        in_pow = (row['studied_credits'].item()*edu/alpha + np.exp(end/10) + np.log(edu)*alpha)/alpha
        # exp = torch.Tensor([[ex_env, ex_phy, in_ach, in_aff, in_pow]])
        exp = torch.Tensor([in_ach, in_aff, in_pow])
        exp = torch.nn.functional.softmax(exp)
        return torch.Tensor([1 if row['assessment_type_CMA'].item() else -1, end*10, age, edu, row['score'].item(), row['num_of_prev_attempts'].item(), row['studied_credits'].item()]), exp

dataset = VLEDataset(df=df)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

network = FNN(n_classes_in=7, n_classes_out=3)
alpha = 10.0
beta = 0.001
n_epoch = 30
lr = 0.01
optimizer = torch.optim.Adam(network.parameters(), lr=lr)
loss_cos = nn.CosineSimilarity(dim=0)
network.train()
for epoch in range(1,n_epoch):
    avg_loss = 0
    prev_loss = 0
    for index, inp in enumerate(dataloader):
        exp = inp[1]
        inp = inp[0]
        out = network(inp)

        # loss = 2 * torch.mean((inp[:,2:]-out)**2) + 0.2 * (3*loss_cos(inp[:,4], out[:,0]) + 2*loss_cos(inp[:,3], out[:,2]) + 1*loss_cos(inp[:,2], out[:,1])) - 0.3*torch.mean((out[:,0]-inp[:,0])**2)
        loss = torch.mean((exp[:,0]-out[:,0])**2) + torch.mean((exp[:,1]-out[:,1])**2) + torch.mean((exp[:,2]-out[:,2])**2)
        # print(out)
        # print(exp)
        avg_loss += loss
        loss.backward()
        optimizer.step()

        print('epoch {}, batches {}, avg_loss = {}'.format(epoch, index, avg_loss.item()-prev_loss))
        prev_loss = avg_loss.item()
    print('epoch {}, avg_loss = {}'.format(epoch, avg_loss.item()/len(dataloader)))
    torch.save(network.state_dict(), './models/model_3_epoch{}.pth'.format(epoch))

torch.save(network.state_dict(), './models/model_3.pth')
