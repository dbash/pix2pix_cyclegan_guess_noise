import torch
from .base_model import BaseModel
from . import networks
import os

class Pix2Segm2PixModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_segm_loss', type=float, default=1.0, help='weight for segm loss')
            parser.add_argument('--segmnets_cpkt_dir', type=str, default='./checkpoints', help='path to G_A2segm and G_B2segm')
        return parser


    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN',  'D_real', 'D_fake', 'segm_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_segm', 'fake_B', 'fake_segm', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'G_A2segm', 'G_B2segm']
        else:  # during test time, only load G
            self.model_names = ['G', 'G_A2segm', 'G_B2segm']
        # define networks (both generator and discriminator)
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG = networks.define_G(input_nc=3, output_nc=3, ngf=64, netG='unet_256',
                                             gpu_ids=self.gpu_ids)
        self.netG_A2segm = networks.define_G(input_nc=3, output_nc=3, ngf=64, netG='unet_256',
                                             gpu_ids=self.gpu_ids)

        self.netG_B2segm = networks.define_G(input_nc=3, output_nc=3, ngf=64, netG='unet_256',
                                             gpu_ids=self.gpu_ids)
        self.pretrained_folder = opt.segmnets_cpkt_dir


        ############## adding pretrained segmentation nets #####################################

        self.load_pretrained_models()

        for param in self.netG_A2segm.parameters(): #fix the weights of the pretrained models
            param.requires_grad = False

        for param in self.netG_B2segm.parameters():
            param.requires_grad = False

        ########################################################################################
        if self.isTrain:  # the discriminator is not conditional
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            #self.criterionL1 = torch.nn.L1Loss()
            self.segm_criterion = torch.nn.L1Loss(reduction='elementwise_mean') # loss for the discrepancies between segmentations
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real_segm = self.netG_A2segm(self.real_A) # S(A)
        self.fake_B = self.netG(self.real_A)  # G(A)
        self.fake_segm = self.netG_B2segm(self.fake_B) # S(G(A))

    def load_pretrained_models(self):
        #### A2segm
        G_A2segm_load_path = os.path.join(self.pretrained_folder, "gta2segm_net_G.pth")
        print("loading pretrained model from %s" % G_A2segm_load_path)
        state_dict = torch.load(G_A2segm_load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self._patch_instance_norm_state_dict(state_dict, self.netG_A2segm.module, key.split('.'))
        self.netG_A2segm.module.load_state_dict(state_dict)
        #### B2segm
        G_B2segm_load_path = os.path.join(self.pretrained_folder, "city2segm_net_G.pth")
        print("loading pretrained model from %s" % G_B2segm_load_path)
        state_dict = torch.load(G_B2segm_load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self._patch_instance_norm_state_dict(state_dict, self.netG_B2segm.module, key.split('.'))
        self.netG_B2segm.module.load_state_dict(state_dict)

        #### Generator
        G_load_path = os.path.join(self.pretrained_folder, "segm2city_net_G.pth")
        print("loading pretrained model from %s" % G_load_path)
        state_dict = torch.load(G_load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self._patch_instance_norm_state_dict(state_dict, self.netG.module, key.split('.'))
        self.netG.module.load_state_dict(state_dict)



    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """

        for name in self.model_names:
            if isinstance(name, str):
                if name == 'G_A2segm':
                    load_filename = "gta2segm_net_G.pth"
                    load_path = os.path.join(self.pretrained_folder, load_filename)
                    print("loading pretrained model from %s" % load_path)
                elif name == 'G_B2segm':
                    load_filename = "city2segm_net_G.pth"
                    load_path = os.path.join(self.pretrained_folder, load_filename)
                    print("loading pretrained model from %s" % load_path)
                else:
                    load_filename = '%s_net_%s.pth' % (epoch, name)
                    load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self._patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        #fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(self.fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        #real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        #fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(self.fake_B)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        #self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        self.loss_segm_L1 = self.segm_criterion(self.real_segm, self.fake_segm)*self.opt.lambda_segm_loss

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_segm_L1 #+ self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

