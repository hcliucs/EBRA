import torch
import torch.nn as nn
import torch.optim as optim
from network import Edge_Extractor, Color_Extractor, Inpaintor, GlobalDis, LocalDis
from loss import AdversarialLoss,PerceptualLoss

class EBRA(nn.Module):
    def __init__(self):
        super(EBRA, self).__init__()
        edge_G = Edge_Extractor()
        color_G = Color_Extractor()
        edge_G.eval()
        color_G.eval()
        inpaintor = Inpaintor()
        glo_D,loc_D = GlobalDis(), LocalDis()
        self.add_module('edge_G', edge_G)
        self.add_module('color_G', color_G)
        self.add_module('inpaintor', inpaintor)
        self.add_module('glo_D', glo_D)
        self.add_module('loc_D', loc_D)
    
        self.iteration = 0
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss()
        perceptual_loss = PerceptualLoss()
        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('perceptual_loss', perceptual_loss)


        LR1=0.0001
        LR2=0.00001
        BETA1=0.0
        BETA2=0.9
       
        
        
        self.G_optimizer = optim.Adam(
            params=self.inpaintor.parameters(),
            lr=float(LR1),
            betas=(BETA1,BETA2)
        )

        self.glo_D_optimizer = optim.Adam(
            params=self.glo_D.parameters(),
            lr=float(LR2),
            betas=(BETA1,BETA2)
        )

        self.loc_D_optimizer = optim.Adam(
            params=self.loc_D.parameters(),
            lr=float(LR1),
            betas=(BETA1,BETA2)
        )


    def process(self, images, mask, bbox_list):
        self.iteration += 1
        
        # zero optimizers
        self.G_optimizer.zero_grad()
        self.glo_D_optimizer.zero_grad()
        self.loc_D_optimizer.zero_grad()
       
        # process outputs
        repaired=self(images,mask)
        gen_loss = 0
        glo_dis_loss = 0
        loc_dis_loss = 0
        
        # discriminator loss
        glo_dis_input_real = images
        glo_dis_input_fake = repaired.detach()
        glo_dis_real = self.glo_D(glo_dis_input_real)        
        glo_dis_fake = self.glo_D(glo_dis_input_fake)        
        glo_dis_real_loss = self.adversarial_loss(glo_dis_real, True, True)
        glo_dis_fake_loss = self.adversarial_loss(glo_dis_fake, False, True)
        glo_dis_loss += (glo_dis_real_loss + glo_dis_fake_loss) / 2

        loc_dis_input_real = local_patch(images,bbox_list)
        loc_dis_input_fake = local_patch(repaired.detach(),bbox_list)
        loc_dis_real = self.loc_D(loc_dis_input_real)        
        loc_dis_fake = self.loc_D(loc_dis_input_fake)        
        loc_dis_real_loss = self.adversarial_loss(loc_dis_real, True, True)
        loc_dis_fake_loss = self.adversarial_loss(loc_dis_fake, False, True)
        loc_dis_loss += (loc_dis_real_loss + loc_dis_fake_loss) / 2


        # generator adversarial loss
        glo_gen_input_fake = repaired
        loc_gen_input_fake = local_patch(repaired,bbox_list)
        glo_gen_fake = self.glo_D(glo_gen_input_fake)     
        loc_gen_fake = self.loc_D(loc_gen_input_fake)     
        glo_gen_gan_loss = self.adversarial_loss(glo_gen_fake, True, False)
        loc_gen_gan_loss = self.adversarial_loss(loc_gen_fake, True, False)
        gen_loss += (glo_gen_gan_loss+loc_gen_gan_loss)

        # generator rec loss
        gen_l1_loss = self.l1_loss(repaired, images)*torch.mean(mask)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(repaired, images)
        gen_content_loss = gen_content_loss * 0.1
        gen_loss += gen_content_loss

        # create logs
        logs = [
            ("l_g", gen_loss.item()),
            ("l_glo_d", glo_dis_loss.item()),
            ("l_loc_d", loc_dis_loss.item()),
            ("l_rec", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            
        ]
        return repaired, gen_loss, glo_dis_loss, loc_dis_loss, logs

    def forward(self, images, mask):
        edge = self.edge_G(images)  
        color = self.color_G(images)
        images_masked = images*(1-mask)
        repaired = self.inpaintor(images_masked,mask,edge,color)
        return repaired

    def backward(self, gen_loss=None, glo_dis_loss=None, loc_dis_loss=None):
        if gen_loss is not None:
            gen_loss.backward()
        self.G_optimizer.step()

        if glo_dis_loss is not None:
            glo_dis_loss.backward()
        self.glo_D_optimizer.step()

        if loc_dis_loss is not None:
            loc_dis_loss.backward()
        self.loc_D_optimizer.step()

    def train(self,dataloader,EPOCH):
        for i in range(EPOCH):
            for items in dataloader:
                img,mask,bbox_list=items
                img=img
                mask=mask
                bbox_list=bbox_list
                repaired, gen_loss, glo_dis_loss, loc_dis_loss, logs = self.process(img,mask,bbox_list)
                self.backward(gen_loss, glo_dis_loss, loc_dis_loss)
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

def local_patch(x, bbox_list):
    assert len(x.size()) == 4
    patches = []
    for bbox in bbox_list:
        t,l,h,w = bbox
        for j in range(t.shape[0]):
            patches.append(x[j, :, t[j]:t[j] + h[j], l[j]:l[j] + w[j]])
    return torch.stack(patches, dim=0)

if __name__=='__main__':
    model=EBRA()