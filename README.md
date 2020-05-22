# L2R-GAN 19/05
1. download 1 dataset finished
#############################
10/01/2019
Time: 12:32:52 GMT
Duration: 00:35:57
Distance: 9.04 km
Total Size: 140.5 GB
Time: 11:46:21 GMT
Duration: 00:37:00
Distance: 9.02 km
Total Size: 143.2 GB


#############################


2. use radar and lidar 401*401 -> should means 40*40 meter  finished
#############################


3. get a fine lidar BEV finished
###############################################################



next -> try to do proprecessing in ccu

###############################################################

next -> train new dataset with pix2pix 256*256 finished -> 200 epcho_19_05


and pix2pixhd in ccu


# L2R-GAN 20/05


###############################################################

ToDo:  train  with pix2pix 512*512 in ccu ->failed
###############################################################

ToDo:  train  with pix2pixHD 400*400 in ccu -> finished
###############################################################

ToDo:  plot and save loss -> finished
###############################################################

L2R-GAN 21/05

######################################################
ToDo :read Augmented LiDAR Simulator for Autonomous Driving

######################################################

ToDo : do the Pix2PixHD test

######################################################

ToDo : think about loss and new network -> loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):



losses, generated = model(Variable(data['label']), Variable(data['inst']), 
            Variable(data['image']), Variable(data['feat']), infer=save_fake)

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)


loss = GAN loss + Feature matching loss + Content loss
Feature matching loss  = perceptual loss
Discriminator loss for Real Samples(D_Real) Discriminator loss for Generated Samples(D_Fake) Generator Least-Squares GAN Loss(G_GAN) Generator Feature Matching Loss(from Discriminator layers) Generator Feature Matching Loss(from Discriminator layers) Optional Generator VGG Perceptual Loss

GAN loss：和pix2pix一样，使用PatchGAN。


Feature matching loss：将生成的样本和Ground truth分别送入判别器提取特征，然后对特征做Element-wise loss


Content loss：将生成的样本和Ground truth分别送入VGG16提取特征，然后对特征做Element-wise loss

ToDo:TensorBoard, input raw data -> ccu 
ToDo:write result and ground truth into html 

ToDo: Improving 3D Object Detection through Progressive Population Based Augmentation

A Method for Synthetic LiDAR Generation to Create Annotated Datasets for Autonomous Vehicles Perception


He's saying reduce the number of nuerons in each layer (settings ngf and ndf) and INCREASE the number of layers (n_blocks_global, n_downsample_global, etc).

I have found by experimentation that finer details come through by increasing, in my case, n_blocks_global from the default 9 to 12 with ngf/ndf 32. Will probably experiment with increasing the number n_blocks_local or going above 12 depending on if the details are good enough.

22/05
-> overfit but loss no down ->why?
