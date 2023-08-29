from .pix2pixHD_model import Pix2PixHDModel
import numpy as np
import torch
import torch.nn as nn
import os
from .base_model import BaseModel
from . import networks

class ESGAN(Pix2PixHDModel):
    def initialize(self, opt):
        super(ESGAN, self).initialize(opt)
        self.criterionRecon = nn.L1Loss()

    def forward(self, label, image, condition):
        real_image = image
        fake_image = self.netG.forward(label, condition)

        # L1 loss
        loss_l1 = self.criterionRecon(fake_image, real_image)

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate(label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(torch.cat((label, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN regression loss
        if self.opt.condition_size:
            regression_fake = pred_fake[0][0][1]
            regression_real = pred_real[0][0][1]
            loss_regress_real = self.criterionAttribute(regression_real, condition)
            loss_regress_fake = self.criterionAttribute(regression_fake, condition)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake[i][j],
                                                          pred_real[i][j].detach()) * self.opt.lambda_feat

        loss_dict = {'L1': loss_l1, 'D_fake': loss_D_fake, 'D_real': loss_D_real, 'G_GAN': loss_G_GAN,
                     'cond_real': loss_regress_real, 'cond_fake': loss_regress_fake}
        return loss_dict, fake_image
    