import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random   

class LogNormalDataset(Dataset):
    def __init__(self, percorso_x, percorso_y):  #, percorso_pk):
        self.campi = torch.load(percorso_x, weights_only=True, mmap=True)
        self.labels = torch.load(percorso_y, weights_only=True)
        # self.pk_data = torch.load(percorso_pk, weights_only=True)
        

        # pre-processing

        campione_statistico = self.campi[:200].flatten().float() # crea array 1D per studiarne la statistica
        
        limite_superiore = torch.quantile(campione_statistico, 0.995)
        limite_inferiore = torch.quantile(campione_statistico, 0.005)
        
        self.campi = torch.clamp(self.campi, min=limite_inferiore, max=limite_superiore)  # evita picchi o valli eccessive per il training
        

        # normalizzazione Z-score

        self.media_x = self.campi.mean()
        self.std_x = self.campi.std()
        self.campi = (self.campi - self.media_x) / self.std_x     
        
        self.media_y = self.labels.mean(dim=0)  # dim=0 per normalizzare i tre parametri singolarmente, colonna per colonna
        self.std_y = self.labels.std(dim=0)
        self.labels = (self.labels - self.media_y) / self.std_y

        # self.pk_data = torch.log10(self.pk_data + 1e-8) # +1e-8 per stabilità numerica
        # self.media_pk = self.pk_data.mean(dim=0)
        # self.std_pk = self.pk_data.std(dim=0)
        # self.pk_data = (self.pk_data - self.media_pk) / self.std_pk

    def __len__(self):
        return len(self.campi)

    def __getitem__(self, idx):
        campo_x = self.campi[idx]
        label_y = self.labels[idx]
        # pk_z = self.pk_data[idx]
        
        # data augmentation
        if random.random() > 0.5:
            campo_x = torch.flip(campo_x, dims=[1]) # rotazione attorno asse orizzontale
            
        if random.random() > 0.5:
            campo_x = torch.flip(campo_x, dims=[2]) # rotazione attorno asse verticale
            
        return campo_x, label_y
        


# Simple CNN

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            

            nn.AdaptiveAvgPool2d((2, 2)), 
            nn.Flatten()
        )


        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )
    
    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x


# VGG

def vgg_block(num_convs, in_channels, out_channels):

    # num_convs convoluzioni 3x3 seguite da un max pooling
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels)) # normalizza dati in uscita
        layers.append(nn.ReLU())    # Leaky per mantenere anche underdensity
        in_channels = out_channels 
        
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    

    return nn.Sequential(*layers)   # spacchetta lista per torch


class CosmoVGG(nn.Module):
    def __init__(self, conv_arch=None):
        super(CosmoVGG, self).__init__()
        
        # architettura di default
        if conv_arch is None:
            conv_arch = (
                (1, 32),    # (num_convs, out_channels)
                (1, 64),   
                (2, 128),  
                (2, 256)    # 4 max pooling -> res_finale=res_iniziale/16
            )
            
        self.features = nn.Sequential()
        
        in_channels = 1
        for i, (num_convs, out_channels) in enumerate(conv_arch):
            block = vgg_block(num_convs, in_channels, out_channels)     
            self.features.add_module(f"vgg_block_{i+1}", block)
            in_channels = out_channels
            
   
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),   # dovrebbe diminuire overfitting e imparare concetti globali; permette anche di accettare qualsiasi dimensione iniziale
            nn.Flatten(),       # pulisce dati into un array 1D
            nn.Dropout(0.4),    # overfitting
            nn.Linear(in_channels, 128),    # fully connected
            nn.ReLU(),
            nn.Linear(128, 3) # output = (omega_m, n_s, sigma_8)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x



# GoogLeNet

class InceptionBlock(nn.Module):

    # 4 rami
    def __init__(self, in_channels, c1, c2, c3, c4):    # c sono canali in uscita di ogni ramo
        super(InceptionBlock, self).__init__()
        
        # RAMO 1: 1x1
        self.ramo1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        
        # RAMO 2: 3x3
        self.ramo2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),   # 1x1 riduce numero canali
            nn.BatchNorm2d(c2[0]),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c2[1]),
            nn.ReLU()
        )
        
        # RAMO 3: 5x5
        self.ramo3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.BatchNorm2d(c3[0]),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(c3[1]),
            nn.ReLU()
        )
        
        # RAMO 4: MaxPool + 1x1
        self.ramo4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.BatchNorm2d(c4),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.ramo1(x)
        out2 = self.ramo2(x)
        out3 = self.ramo3(x)
        out4 = self.ramo4(x)      
        
        return torch.cat((out1, out2, out3, out4), dim=1)   # concatena le feature lungo la dimensione dei canali (dim=1)


class CosmoGoogLeNet(nn.Module):
    def __init__(self):
        super(CosmoGoogLeNet, self).__init__()
        
        self.stem = nn.Sequential(      # res_finale = res_iniziale/4
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),   # dimezza risoluzione
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)    # dimezza risoluzione
        )
        
      
        
        # Inception 1: Entrano 32, escono 16 + 64 + 16 + 16 = 112 canali
        self.inc1 = InceptionBlock(32, 16, (32, 64), (8, 16), 16)
        
        # Inception 2: Entrano 112, escono 32 + 128 + 32 + 32 = 224 canali
        self.inc2 = InceptionBlock(112, 32, (64, 128), (16, 32), 32)
        
        self.pool_mezzo = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # -> # res_finale = res_iniziale/8
        
        # Inception 3: Entrano 224, escono 64 + 256 + 64 + 64 = 448 canali
        self.inc3 = InceptionBlock(224, 64, (128, 256), (32, 64), 64)
        

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(448, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # output = (omega_m, n_s, sigma_8)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.pool_mezzo(x)
        x = self.inc3(x)
        x = self.head(x)
        return x




# ResNet

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super(ResidualBlock, self).__init__()
        
       
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,                # questa conv può cambiare res
                               padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky = nn.ReLU()


        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)    # questa mantiene sempre stessa res
        self.bn2 = nn.BatchNorm2d(out_channels)
        

        if use_1x1conv:                                                                 # serve per adattare dimensioni quando sommo nel forward
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, X):
        Y = self.leaky(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        
        Y = Y + self.shortcut(X)
        
        return self.leaky(Y)



def crea_fase_resnet(in_channels, out_channels, num_blocks, prima_fase=False):

    blocks = []
    for i in range(num_blocks):
        if i == 0 and not prima_fase:
            blocks.append(ResidualBlock(in_channels, out_channels, use_1x1conv=True, strides=2))
        elif i == 0 and prima_fase and in_channels != out_channels:
            blocks.append(ResidualBlock(in_channels, out_channels, use_1x1conv=True, strides=1))
        else:
            blocks.append(ResidualBlock(out_channels, out_channels))
            
    return nn.Sequential(*blocks)


class CosmoResNet(nn.Module):
    def __init__(self, architettura=((2, 64), (2, 128), (2, 256), (2, 512))):   # ResNet-18
        super(CosmoResNet, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.fasi = nn.Sequential()
        in_channels = 64    # quelli che escono dallo stem
        
        for i, (num_blocks, out_channels) in enumerate(architettura):
            is_prima_fase = (i == 0) 
            
            fase = crea_fase_resnet(in_channels, out_channels, num_blocks, prima_fase=is_prima_fase)
            
            self.fasi.add_module(f"fase_resnet_{i+1}", fase)
            
            in_channels = out_channels
            
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # output = (omega_m, n_s, sigma_8)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.fasi(x)
        x = self.head(x)
        return x



class CosmoResNet3head(nn.Module):
    def __init__(self, architettura=((2, 64), (2, 128), (2, 256), (2, 512))):   # ResNet-18
        super(CosmoResNet3head, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.fasi = nn.Sequential()
        in_channels = 64    # quelli che escono dallo stem
        
        for i, (num_blocks, out_channels) in enumerate(architettura):
            is_prima_fase = (i == 0) 
            
            fase = crea_fase_resnet(in_channels, out_channels, num_blocks, prima_fase=is_prima_fase)
            
            self.fasi.add_module(f"fase_resnet_{i+1}", fase)
            
            in_channels = out_channels
            
        self.pool= nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
        )

        self.head_om = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

        self.head_ns = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

        self.head_s8 = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.fasi(x)
        x = self.pool(x)
        out_om = self.head_om(x)
        out_ns = self.head_ns(x)
        out_s8 = self.head_s8(x)
        return torch.cat([out_om, out_ns, out_s8], dim=1)



class CosmoResNetNLL(nn.Module):
    def __init__(self, architettura=((2, 64), (2, 128), (2, 256), (2, 512))):   # ResNet-18      (3, 64), (4, 128), (6, 256), (3, 512)
        super(CosmoResNetNLL, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.fasi = nn.Sequential()
        in_channels = 64    # quelli che escono dallo stem
        
        for i, (num_blocks, out_channels) in enumerate(architettura):
            is_prima_fase = (i == 0) 
            
            fase = crea_fase_resnet(in_channels, out_channels, num_blocks, prima_fase=is_prima_fase)
            
            self.fasi.add_module(f"fase_resnet_{i+1}", fase)
            
            in_channels = out_channels
            
        self.pool= nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
        )

        self.head_om = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 2) 
        )

        self.head_ns = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 2) 
        )

        self.head_s8 = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 2) 
        )

        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.stem(x)
        x = self.fasi(x)
        x = self.pool(x)
        out_om = self.head_om(x)
        out_ns = self.head_ns(x)
        out_s8 = self.head_s8(x)

        mu_om, var_om = out_om[:, 0:1], self.softplus(out_om[:, 1:2]) + 1e-6
        mu_ns, var_ns = out_ns[:, 0:1], self.softplus(out_ns[:, 1:2]) + 1e-6
        mu_s8, var_s8 = out_s8[:, 0:1], self.softplus(out_s8[:, 1:2]) + 1e-6

        medie = torch.cat([mu_om, mu_ns, mu_s8], dim=1)
        varianze = torch.cat([var_om, var_ns, var_s8], dim=1)

        return medie, varianze





class SimpleIbridaCosmoNet(nn.Module):
    def __init__(self, pk_length):
        super(SimpleIbridaCosmoNet, self).__init__()
        
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            

            nn.AdaptiveAvgPool2d((4, 4)), 
            nn.Flatten()
        )

        self.pk_branch = nn.Sequential(
            nn.Linear(pk_length, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, img, pk):
        img_features = self.cnn_branch(img)
        pk_features = self.pk_branch(pk)
        
        # concatenazione lungo la dimensione delle feature (dim=1, poiché dim=0 è il batch)
        combined_features = torch.cat((img_features, pk_features), dim=1)
        
        output = self.head(combined_features)
        
        return output



class IbridaCosmoResNet(nn.Module):
    def __init__(self, pk_length, architettura=((2, 64), (2, 128), (2, 256), (2, 512))):
        super(IbridaCosmoResNet, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.fasi = nn.Sequential()
        in_channels = 64    # quelli che escono dallo stem
        
        for i, (num_blocks, out_channels) in enumerate(architettura):
            is_prima_fase = (i == 0) 
            
            fase = crea_fase_resnet(in_channels, out_channels, num_blocks, prima_fase=is_prima_fase)
            
            self.fasi.add_module(f"fase_resnet_{i+1}", fase)
            
            in_channels = out_channels
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()

        self.pk_branch = nn.Sequential(
            nn.Linear(pk_length, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_channels + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, img, pk):
        img_features = self.flatten(self.pool(self.fasi(self.stem(img))))
        pk_features = self.pk_branch(pk)
        
        # concatenazione lungo la dimensione delle feature (dim=1, poiché dim=0 è il batch)
        combined_features = torch.cat((img_features, pk_features), dim=1)
        
        output = self.head(combined_features)
        
        return output
