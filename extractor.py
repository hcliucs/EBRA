import os
import torch
import torch.nn as nn
import torch.optim as optim
from network import Edge_Extractor, Color_Extractor, Discriminator
from loss import AdversarialLoss



class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        edge_G = Edge_Extractor()
        color_G = Color_Extractor()
        edge_D = Discriminator(in_channels=1)
        color_D = Discriminator(in_channels=3)
        self.add_module('edge_G', edge_G)
        self.add_module('color_G', color_G)
        self.add_module('edge_D', edge_D)
        self.add_module('color_D', color_D)
        
        self.iteration = 0
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss()
        LR1=0.0001
        LR2=0.00001
        BETA1=0.0
        BETA2=0.9
        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.edge_G_optimizer = optim.Adam(
            params=edge_G.parameters(),
            lr=float(LR1),
            betas=(BETA1,BETA2)
        )

        self.edge_D_optimizer = optim.Adam(
            params=edge_D.parameters(),
            lr=float(LR2),
            betas=(BETA1,BETA2)
        )

        self.color_G_optimizer = optim.Adam(
            params=color_G.parameters(),
            lr=float(LR1),
            betas=(BETA1,BETA2)
        )

        self.color_D_optimizer = optim.Adam(
            params=color_D.parameters(),
            lr=float(LR2),
            betas=(BETA1,BETA2)
        )

    def process(self, images, edge_ground, color_ground):
       
        self.iteration += 1
        # zero optimizers
        self.color_G_optimizer.zero_grad()
        self.color_D_optimizer.zero_grad()
        self.edge_G_optimizer.zero_grad()
        self.edge_D_optimizer.zero_grad()
        # process outputs
        edge, color = self(images)
        edge_gen_loss = 0
        edge_dis_loss = 0
        color_gen_loss = 0
        color_dis_loss = 0

        # discriminator loss
        edge_dis_input_real = edge_ground
        edge_dis_input_fake = edge.detach()
        edge_dis_real, edge_dis_real_feat = self.edge_D(edge_dis_input_real)        
        edge_dis_fake, _ = self.edge_D(edge_dis_input_fake)        
        edge_dis_real_loss = self.adversarial_loss(edge_dis_real, True, True)
        edge_dis_fake_loss = self.adversarial_loss(edge_dis_fake, False, True)
        edge_dis_loss += (edge_dis_real_loss + edge_dis_fake_loss) / 2

        color_dis_input_real = color_ground
        color_dis_input_fake = color.detach()
        color_dis_real, color_dis_real_feat = self.color_D(color_dis_input_real)        
        color_dis_fake, _ = self.color_D(color_dis_input_fake)        
        color_dis_real_loss = self.adversarial_loss(color_dis_real, True, True)
        color_dis_fake_loss = self.adversarial_loss(color_dis_fake, False, True)
        color_dis_loss += (color_dis_real_loss + color_dis_fake_loss) / 2

        # generator adversarial loss
        edge_gen_input_fake = edge
        edge_gen_fake, edge_gen_fake_feat = self.edge_D(edge_gen_input_fake)        
        edge_gen_gan_loss = self.adversarial_loss(edge_gen_fake, True, False)
        edge_gen_loss += edge_gen_gan_loss

        color_gen_input_fake = color
        color_gen_fake, color_gen_fake_feat = self.color_D(color_gen_input_fake)        
        color_gen_gan_loss = self.adversarial_loss(color_gen_fake, True, False)
        color_gen_loss += color_gen_gan_loss

        # generator feature matching loss
        edge_gen_fm_loss = 0
        for i in range(len(edge_dis_real_feat)):
            edge_gen_fm_loss += self.l1_loss(edge_gen_fake_feat[i], edge_dis_real_feat[i].detach())
        edge_gen_fm_loss = edge_gen_fm_loss * 10
        edge_gen_loss += edge_gen_fm_loss

        color_gen_fm_loss = 0
        for i in range(len(color_dis_real_feat)):
            color_gen_fm_loss += self.l1_loss(color_gen_fake_feat[i], color_dis_real_feat[i].detach())
        color_gen_fm_loss = color_gen_fm_loss * 10
        color_gen_loss += color_gen_fm_loss
        
        # create logs
        logs = [
            ("edge_d", edge_dis_loss.item()),
            ("edge_g", edge_gen_gan_loss.item()),
            ("edge_fm", edge_gen_fm_loss.item()),
            ("color_d", color_dis_loss.item()),
            ("color_g", color_gen_gan_loss.item()),
            ("color_fm", color_gen_fm_loss.item()),
        ]
        return edge, color, edge_gen_loss, edge_dis_loss, color_gen_loss, color_dis_loss, logs

    def forward(self, images):
        edge = self.edge_G(images)  
        color = self.color_G(images)
        return edge[3],color[3]

    def backward(self, edge_gen_loss=None, edge_dis_loss=None, color_gen_loss=None, color_dis_loss=None):
        if edge_dis_loss is not None:
            edge_dis_loss.backward()
        self.edge_D_optimizer.step()

        if edge_gen_loss is not None:
            edge_gen_loss.backward()
        self.edge_G_optimizer.step()

        if color_dis_loss is not None:
            color_dis_loss.backward()
        self.color_D_optimizer.step()

        if color_gen_loss is not None:
            color_gen_loss.backward()
        self.color_G_optimizer.step()

    def train(self,dataloader,EPOCH):
        for i in range(EPOCH):
            for items in dataloader:
                img,edge_T,color_T=items
                img=img
                edge_T=edge_T
                color_T=color_T

                edge, color, edge_gen_loss, edge_dis_loss, color_gen_loss, color_dis_loss, logs = self.process(img,edge_T,color_T)
                self.backward(edge_gen_loss, edge_dis_loss, color_gen_loss, color_dis_loss)
        self.save()
    
    def save(self):
        torch.save({
            'iteration': self.iteration,
            'edge': self.edge_G.state_dict(),
            'color': self.color_G.state_dict()
        }, './checkpoints/edge_color_G.pth')

        torch.save({
            'edge': self.edge_D.state_dict(),
            'color': self.color_D.state_dict()
        }, './checkpoints/edge_color_D.pth')
   