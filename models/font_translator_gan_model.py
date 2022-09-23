import torch
from .base_model import BaseModel
from . import networks


class FontTranslatorGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):  # 修改命令行的选项
        # changing the default values
        # parser.set_defaults(norm='batch', netG='FTGAN_MLAN',
        #                     dataset_mode='font')  # 设置默认属性
        parser.set_defaults(norm='batch', netG='FTGAN_MLAN')  # 设置默认属性
        if is_train:
            parser.set_defaults(batch_size=64, pool_size=0,
                                gan_mode='hinge', netD='basic_64')  # 64个batchsize
            parser.add_argument('--lambda_L1', type=float,
                                default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_style', type=float,
                                default=1.0, help='weight for style loss')
            parser.add_argument('--lambda_content', type=float,
                                default=1.0, help='weight for content loss')
            parser.add_argument('--dis_2', default=True,
                                help='use two discriminators or not')  # 使用两个鉴别器
            parser.add_argument('--use_spectral_norm', default=True)
        return parser

    def __init__(self, opt):
        """Initialize the font_translator_gan class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)  # 初始化
        self.style_channel = opt.style_channel  # 参考的风格图片

        if self.isTrain:
            self.dis_2 = opt.dis_2  # 绘制虚拟
            self.visual_names = ['gt_images', 'generated_images'] + \
                ['style_images_{}'.format(i)
                 for i in range(self.style_channel)]
            if self.dis_2:
                self.model_names = ['G', 'D_content',
                                    'D_style']  # 然后一个生成，一个区分内容，一个区分风格
                self.loss_names = ['G_GAN', 'G_L1',
                                   'D_content', 'D_style']  # 计算loss的模块
            else:
                self.model_names = ['G', 'D']  # 其他的话直接按整体进行gan
                self.loss_names = ['G_GAN', 'G_L1', 'D']
        else:
            self.visual_names = ['gt_images',
                                 'generated_images']  # 如果是测试直接就是生成模块
            self.model_names = ['G']
        # define networks (both generator and discriminator) #最后一层输出通道数，resnet9block
        self.netG = networks.define_G(self.style_channel+1, 1, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)  # 定义网络,resnet_9blocks,bn,nodropout,
        # 如果是train就加上d内容 和风格
        if self.isTrain:  # define discriminators; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            if self.dis_2:
                self.netD_content = networks.define_D(2, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type,
                                                      opt.init_gain, self.gpu_ids, use_spectral_norm=opt.use_spectral_norm)  # 内容生成器 一个是2
                self.netD_style = networks.define_D(self.style_channel+1, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                                    opt.init_type, opt.init_gain, self.gpu_ids, use_spectral_norm=opt.use_spectral_norm)  # 类别风格生成器 一个是六+1
            else:
                self.netD = networks.define_D(self.style_channel+2, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                              opt.init_type, opt.init_gain, self.gpu_ids, use_spectral_norm=opt.use_spectral_norm)

        if self.isTrain:
            # define loss functions
            self.lambda_L1 = opt.lambda_L1
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode).to(self.device)  # 标准评判,使用gpu
            self.criterionL1 = torch.nn.L1Loss()  # 再用l1loss计算两个元素之间的绝对误差
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(
            ), lr=opt.lr, betas=(opt.beta1, 0.999))  # 将定义完的网络放入优化器优化
            self.optimizers.append(self.optimizer_G)
            if self.dis_2:
                self.lambda_style = opt.lambda_style  # 权重值 1.0
                self.lambda_content = opt.lambda_content  # 权重值 1.0
                self.optimizer_D_content = torch.optim.Adam(self.netD_content.parameters(
                ), lr=opt.lr, betas=(opt.beta1, 0.999))  # 优化器学习率推进把D也推进去，第一次衰减率
                self.optimizer_D_style = torch.optim.Adam(
                    self.netD_style.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                # self.optimizer_D_content = torch.optim.AdamW(self.netD_content.parameters(), lr=opt.lr_adamw)#优化器学习率推进把D也推进去，第一次衰减率
                # self.optimizer_D_style = torch.optim.AdamW(self.netD_style.parameters(),lr=opt.lr_adamw)#
                self.optimizers.append(self.optimizer_D_content)  # 优化器添加进去
                self.optimizers.append(self.optimizer_D_style)  # 再加上风格
            else:
                self.optimizer_D = torch.optim.Adam(
                    self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            # 初始化完毕

    def set_input(self, data):
        if self.isTrain:
            self.gt_images = data['gt_images'].to(self.device)
        self.content_images = data['content_images'].to(self.device)
        self.style_images = data['style_images'].to(self.device)
        if not self.isTrain:
            self.image_paths = data['image_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.generated_images = self.netG(
            (self.content_images, self.style_images))  # 内容图片，风格图片

    def compute_gan_loss_D(self, real_images, fake_images, netD):
        # Fake
        fake = torch.cat(fake_images, 1)  # 扩张一维
        pred_fake = netD(fake.detach())  # 不参加参数更新
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real = torch.cat(real_images, 1)  # 混合放进去
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5  # 两个loss加权*0.5
        return loss_D

    def compute_gan_loss_G(self, fake_images, netD):
        fake = torch.cat(fake_images, 1)
        pred_fake = netD(fake)
        loss_G_GAN = self.criterionGAN(pred_fake, True, True)
        return loss_G_GAN

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        if self.dis_2:  # 内容计算loss，真实图片与内容图片计算，生成图片与内容计算
            self.loss_D_content = self.compute_gan_loss_D([self.content_images, self.gt_images],  [
                                                          self.content_images, self.generated_images], self.netD_content)
            self.loss_D_style = self.compute_gan_loss_D([self.style_images, self.gt_images], [
                                                        self.style_images, self.generated_images], self.netD_style)  # 同理
            self.loss_D = self.lambda_content*self.loss_D_content + \
                self.lambda_style*self.loss_D_style
        else:
            self.loss_D = self.compute_gan_loss_D([self.content_images, self.style_images, self.gt_images], [
                                                  self.content_images, self.style_images, self.generated_images], self.netD)

        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.dis_2:
            self.loss_G_content = self.compute_gan_loss_G(
                [self.content_images, self.generated_images], self.netD_content)
            self.loss_G_style = self.compute_gan_loss_G(
                [self.style_images, self.generated_images], self.netD_style)
            self.loss_G_GAN = self.lambda_content * \
                self.loss_G_content + self.lambda_style*self.loss_G_style
        else:
            self.loss_G_GAN = self.compute_gan_loss_G(
                [self.content_images, self.style_images, self.generated_images], self.netD)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(
            self.generated_images, self.gt_images) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):  # 优化器
        self.forward()                   # compute fake images: G(A)
        # update D
        if self.dis_2:  # 两次分辨
            self.set_requires_grad([self.netD_content, self.netD_style], True)
            self.optimizer_D_content.zero_grad()  # 梯度清除
            self.optimizer_D_style.zero_grad()
            self.backward_D()  # 计算loss，反向传播
            self.optimizer_D_content.step()  # 使用梯度
            self.optimizer_D_style.step()
        else:
            # enable backprop for D
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()             # set D's gradients to zero
            self.backward_D()                    # calculate gradients for D
            self.optimizer_D.step()                # update D's weights
        # update G
        if self.dis_2:
            self.set_requires_grad([self.netD_content, self.netD_style], False)
        else:
            # D requires no gradients when optimizing G
            self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()                  # set G's gradients to zero
        self.backward_G()                             # calculate graidents for G
        self.optimizer_G.step()                       # udpate G's weights

    def compute_visuals(self):
        if self.isTrain:
            self.netG.eval()  # 只生
            with torch.no_grad():
                self.forward()
            for i in range(self.style_channel):  # 遍历扩容
                setattr(self, 'style_images_{}'.format(i),
                        torch.unsqueeze(self.style_images[:, i, :, :], 1))
            self.netG.train()  # 开始训练
        else:
            pass
