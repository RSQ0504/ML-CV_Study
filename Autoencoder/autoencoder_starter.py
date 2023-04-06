
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import imageio
import matplotlib.image as mpimg
from scipy import ndimage



def scatter_plot(latent_representations, labels):
    '''
    the scatter plot for visualizing the latent representations with the ground truth class label
    ----------
    latent_presentations: (N, dimension_latent_representation)
    labels: (N, )  the labels of the ground truth classes
    '''
    # borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
    def discrete_cmap(n, base_cmap=None):
        """Create an N-bin discrete colormap from the specified input map"""
        base = plt.cm.get_cmap(base_cmap)
        return base.from_list(base.name + str(n), base(np.linspace(0, 1, n)), n)

    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0], latent_representations[:, 1], cmap=discrete_cmap(10, 'jet'), c=labels, edgecolors='black')
    plt.colorbar()
    plt.grid()
    plt.show()

def display_images_in_a_row(images,file_path='./tmp.png', display=True):
  '''
  images: (N,28,28): N images of 28*28 as a numpy array
  file_path: file path name for where to store the figure
  display: display the image or not
  '''
  save_image(images.view(-1, 1, 28, 28),
            '{}'.format(file_path))
  if display is True:
    plt.imshow(mpimg.imread('{}'.format(file_path)))


# Defining Model
class Autoencoder_Trainer(object):
    '''
    The trainer for
    '''
    def __init__(self, autoencoder_model, learning_rate=1e-3, path_prefix = ""):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_dataset(path_prefix)
        self.model = autoencoder_model
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)


    def init_dataset(self, path_prefix = ""):
        # load and preprocess dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainTransform  = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.MNIST(root='{}/./data'.format(path_prefix),  train=True,download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=4)
        valset = torchvision.datasets.MNIST(root='{}/./data'.format(path_prefix), train=False, download=True, transform=transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False, num_workers=4)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.valset = valset
        self.trainset = trainset

    def loss_function(self, recon_x, x):
        BCE = F.mse_loss(recon_x, x)
        return BCE
	
    def get_train_set(self):
        images = torch.vstack([ x for x,_ in self.train_loader]) # get the entire train set
        return images
        
    def get_val_set(self):
        images = torch.vstack([ x for x,_ in self.val_loader]) # get the entire val set
        return images
    
    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in tqdm(enumerate(self.train_loader), total=len(self.train_loader) ) :
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch = self.model(data)
            loss = self.loss_function(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        train_loss /= len(self.train_loader.dataset)/32 # 32 is the batch size
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss ))

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (data, _) in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                data = data.to(self.device)
                recon_batch = self.model(data)
                val_loss += self.loss_function(recon_batch, data).item()

        val_loss /= len(self.val_loader.dataset)/32 # 32 is the batch size
        print('====> Val set loss (reconstruction error) : {:.4f}'.format(val_loss))
