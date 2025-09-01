import torch
from torch import nn
import torch.nn.functional as F


class NanoDiffusionTrainer:
    """Trainer for the Nano Diffusion model

    TODO:
    Add all administrative stuff in the Trainer class, e.g. logging, saving model 
    checkpoints etc., and everything concerning the training run in the train function. 
    """

    def __init__(self, model, vae, sampler, optimizer, noise_scheduler):
        self.model = model
        self.sampler = sampler
        self.optimizer = optimizer
        self.noise_scheduler = noise_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader


    def train(self, epochs, optimizer, lr_scheduler, train_dataloader, val_dataloader=None):
        for epoch in range(epochs):
            for latents, context in self.train_dataloader: