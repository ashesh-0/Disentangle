# Objective

Here, our goal is to disentangle the images by taking as input the sum of the two images. For example, given image A &
B, input to our model is A+B and the output is A,B

Example run:
python disentangle/scripts/run.py --workdir=/home/ubuntu/ashesh/training/disentangle/ -mode=train
--datadir=/mnt/ashesh/places365_noisy/
--config=/home/ubuntu/ashesh/code/Disentangle/disentangle/configs/places_lvae_twindecoder_config.py

## Exp 1

Tried two samplers: Random sampler and SingleImg sampler. In first, a batch comprises of two random samples. (
/home/ubuntu/ashesh/training/disentangle/2202/D2-M3-S2-L0/0)
In second a batch comprises of 1 image and other random samples. (
/home/ubuntu/ashesh/training/disentangle/2202/D2-M3-S2-L0/1)
I see that the second approach had better results. however, in both cases, best performance is achived at epoch 1.
which is very strange?

There was a bug in the validation error due to which I'm re-running the configs.
/home/ubuntu/ashesh/training/disentangle/2202/D2-M3-S1-L0/1: Random sampler
/home/ubuntu/ashesh/training/disentangle/2202/D2-M3-S2-L0/5: Single img sampler

Random sampler is better in terms of performance

[airfield                  ice_skating_rink-outdoor ] 50.1 +- 0.4
[grotto                    heliport                 ] 50.3 +- 0.3
[movie_theater-indoor      mezzanine                ] 50.9 +- 0.6
[wet_bar                   movie_theater-indoor     ] 50.8 +- 0.5
[ice_skating_rink-outdoor  waiting_room             ] 51.5 +- 0.4

Increasing the power of reconstruction
3 layered.

DIR = /home/ubuntu/ashesh/training/

Varying (model.z_dims, Val Metrics Train Metrics
Num of filters)
ruth:   (32,16)     Rec:0.0153 KL:0.0030 Rec:0.014404 KL:0.002947 disentangle/2203/D1-M3-S1-L0/1
turing: (128, 64)   Rec:0.0104 KL:0.0027 Rec:0.007735 KL:0.002670 disentangle/2203/D1-M3-S1-L0/1
turing: (256,128)   Rec:0.0091 KL:0.0021 Rec:0.005894 KL:0.002202 disentangle/2203/D1-M3-S1-L0/2

Varying kl_weight
Server kl_weight Val Metrics Train Metrics Ckpt
tur:    1 Rec:0.0104 KL:0.0027 Rec:0.007735 KL:0.002670 disentangle/2203/D1-M3-S1-L0/1
ruth:   0.5 Rec:0.0099 KL:0.0044 Rec:0.007674 KL:0.004390 disentangle/2203/D1-M3-S1-L0/2
tur:    0.2 Rec:0.0089 KL:0.0076 Rec:0.005330 KL:0.007597 disentangle/2203/D1-M3-S1-L0/3
ruth:   0.1 Rec:0.0087 KL:0.0115 Rec:0.004557 KL:0.011589 disentangle/2203/D1-M3-S1-L0/3
tur:    0.05 Rec:0.0088 KL:0.0174 Rec:0.006042 KL:0.017418 disentangle/2203/D1-M3-S1-L0/4
tur:    0.01 Rec:0.0082 KL:0.0332 Rec:0.003737 KL:0.033354 disentangle/2203/D1-M3-S1-L0/5

I observe that the validation is not robust enough as the number of samples are less.
One can easily fix this by increasing the random pairs. I checked it in validation. it is not much significant.

blocks per layer =5
tur:                    Rec:0.007762 KL:0.016818 Rec:0.005100 KL:0.017000 disentangle/2204/D1-M3-S1-L0/0

z_dims from 3 to 2
ruth:   2 Rec:0.0087 KL:0.0209 Rec:0.0064 KL:0.0206 disentangle/2204/D1-M3-S1-L0/0
tur:    3 Rec:0.0088 KL:0.0174 Rec:0.0060 KL:0.0174 disentangle/2203/D1-M3-S1-L0/4

I see strong overfitting when kl_weight=0.01. Adding dropout to see if it improves performance.
Server dropout Val Metrics Train Metrics Ckpt
ruth:   0.2 Rec:0.0081 KL:0.0400 Rec:0.0054 KL:0.0399 disentangle/2204/D1-M3-S1-L0/1
tur:    0.0 Rec:0.0082 KL:0.0332 Rec:0.0037 KL:0.0334 disentangle/2203/D1-M3-S1-L0/5

I checked that KL divergence calculation is fine in the code.
Twindecoder:    
Server Config Val Metrics
tur kl_weight: 0.1 Rec:0.008110 KL:0.007804 Rec:0.006085 KL:0.007914 disentangle/2204/D1-M4-S1-L0/0
ruth kl_weight: 0.01 Rec:0.007787 KL:0.026159 Rec:0.005435 KL:0.026358 disentangle/2204/D1-M4-S1-L0/0
In order to overfit most info, I've reduced the kl_weight to 0.0001
ruth /home/ubuntu/ashesh/training/disentangle/2204/D1-M4-S1-L0/1

# Working with microscopy data

Now, I'm working with OptiMEM100x014 dataset. It has total of 61 images.
I see that val loss shoots to infinity?
Trying a 0.995 quantile as the higher threshold. I see that both image types have same threshold value.

There is asymmetry in the losses of the two classes. one class has more data. other class has less data.
So, the loss is mostly dominated by the one class.

/home/ubuntu/ashesh/training/disentangle/2204/D3-M3-S0-L0/2 is the first working model.
Notes:
It estimates one class better than other. Better performing class typically has more pixels with a pattern.
To fix this, I've code up a hack. Here, one pair of images is valid only if both of them have content.
In the previous model, if one of them had a content, then we considered it a valid pair. That approach is more real
world aligned. However, for the heck of trying it out, I'm trying it to see whether I get better performance
on such images.

VAL Error
/home/ubuntu/ashesh/training/disentangle/2204/D3-M3-S0-L0/4: 0.017
/home/ubuntu/ashesh/training/disentangle/2204/D3-M3-S0-L0/3: 0.009
/home/ubuntu/ashesh/training/disentangle/2204/D3-M3-S0-L0/2: 0.009
/home/ubuntu/ashesh/training/disentangle/2204/D3-M3-S0-L0/6: 0.010

I see that one issue could be simply too less validation samples taken for the loss. Due to this,
I think I was seeing very low losses while training (on evaluation). Which were not reproducible.
With repeating 50 times, I'm able to get reasonable numbers.
/home/ubuntu/ashesh/training/disentangle/2204/D3-M3-S0-L0/7
I don't see validation error stablizing at all.

I think one may repeat the training factor to 10 times as well. This will help in ensuring that most time is not spent
on running the validation only.
Increased the batch_size to 8 and introduced the training factor.
/home/ubuntu/ashesh/training/disentangle/2204/D3-M3-S0-L0/8

To ascertain that the discripency in the validation error is just in due to the fact that we have too few of images and
random sampling of patches leads to the discripency, I've created a deterministic data loader. It samples all the grid
patches. So the validation error should be deterministic now.
/home/ubuntu/ashesh/training/disentangle/2204/D3-M3-S0-L0/9

I found out the bug. It was that the mean() and std() on training data was different than on validation data.
In the notebook I was using the validation data's mean and std due to which this was happening. Now, the validation
performance while training and the performance in notebook matches very closely.

I tried to sample a reconstruction multiple times and then take the average. However, it has no real benefit as the loss
decreases only by 10e-4 units. If I increase the KL weight, then it may become more significant.

/home/ubuntu/ashesh/training/disentangle/
config with deterministic randomized
2204/D3-M3-S0-L0/9 trained with determ 0.010+-0.001
2204/D3-M3-S0-L0/7 trained with random 0.020+-0.002
2204/D3-M3-S0-L0/10 trained with random

## May 7

I see that the mean and the std of the two channels are different. It therefore makes sense to have different mean and
std for normalization.

Using different mean and stdev for the two channels.:
/home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/1
I see that it is not giving a good performance. If this is just because the separate mean std doesn't work,
then passing the original value of the mean and std should lead to identical performance. Working for that.

same mean and std: /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/2
todo: skip the mean/std computtaion everytime => DONE
VAL loss seems to be quite fluctuating. It could be so because of either of the two issues:

1. The threshold is inappropriately high and so training is getting slightly skewed. To test this, I probably just need
   to reduce the threshold to 0 and check.

2. Everytime one gets a different validation set due to random sampling and so we observe very different results. Since
   we are using quite high repeat_factor, I don't think that would be the case. To check if this is the case, I could
   simply evaluate it to see how different are the results that I'm getting.
   Computing the reconstruction loss 3 times.
   Rec:0.051260 KL:nan
   Rec L1:0.029837 Rec L2:0.072700
   Rec:0.052363 KL:nan
   Rec L1:0.027938 Rec L2:0.076840
   Rec:0.051050 KL:nan
   Rec L1:0.029995 Rec L2:0.072093
   As I see, this is sufficiently stable even with val_repeat_factor being just 10. So, reason has to be 1.

   I see that L2 has a much higher loss.
   When I disable the normalized_input, I see that I'm getting a much better loss
   Rec:0.018831 KL:nan
   Rec L1:0.016865 Rec L2:0.020857

I've figured out what the issue was. When normalizing in the data loader, I need to always use the mean and std of
training data and not the validation data. Currently, I was using mean and std of validation data for validation data
loader and mean and std of training data.

/home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/4: This is now behaving just the way when normalized_input was
False.

/home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/5: enabled differing mean and std. Only issue is that it
crashed after sometime. running it again. /6. It still crashes. I'll now try the same idea with deterministic_grid=True

/home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/7: rutherford where I've modified the code to have same mean
and std
/home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/0: turing where I've undoed the changes in the code. The mean
and std will have a different value now.

Both of them are working perfectly well. What this means is that the determinstic grid is ensuring that training happens
as it should happen.

turing: /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/1 prior fixed to N(0,1) and using different mean and
var. It crashed. ? No idea why it crashed.
ruth: /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/8 prior fixed to N(0,1) and using same mean and var
I get the same result. this variable does not seem to have any effect. The reason could be that we are using ladder VAE
and so only the highest layer has the prior fixed to N(0,1). For The lower layers, anyways the convolution filters come
in and they ensure that the p() could be whatever normal distribution. So, it is no wonder that it does not help with
crashing and it did not change the performance.

While doing these things, it makes sense to disable stochastic_skip as well and see what effect does it has
ruth /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/9: stochastic_skip=False => performance is quite similar

With MSE, I see that the best cases are mainly those where there is very little content. In that sense PSNR might be a
better metric to observe things.

To fix the variance exploding problem I've added a maximum value limit on the variance. I simply use clipping.
tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/3

/home/ubuntu/ashesh/training/disentangle/2205/D3-M5-S0-L1/0
Here

/home/ubuntu/ashesh/training/disentangle/2205/D3-M5-S0-L1/1
Here, increased the critic weight 0.1. What happened is the opposite effect. Earlier, both generated and target images
were giving roughly similar predictions in the discriminator. However, now, the generated images have much higher values
than what the target images have. As far as updating the weights are concerned, they are geting updated correctly. I see
that after optimizer 1 runs, the weights are updated in the VAE model. After optimizer 2 runs, weights are updated for
the D model.
I've couple of things in mind which I'll try:

1. Increase the learning rate for the second optimizer. (ruth
   /home/ubuntu/ashesh/training/disentangle/2205/D3-M5-S0-L1/0)
2. Disable the critic loss in the first optimizer. In this case the model should give clear 0/1 predictions:
   0 for the generated and 1 for the target. (tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M5-S0-L1/2)
   It is not giving very clear signals for the second label. And for the first label too, I don't see it going to high
   probablities like 1. It hovers around 0.3-0.5

#TODO check if clipping the var leads to inferior performance. (ruth
/home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/10) => no, it does not

#TODO check if different mean and var with clipping on leads to better peformance.

Added logging in the critic's output to check what is happeniing. also enabled dense layers (tur
/home/ubuntu/ashesh/training/disentangle/2205/D3-M5-S0-L1/3)

# May 17

One thing is that the discriminator needs to be created based on local'ish' features and not super global features. It
therefore does not make sense to have a pretty deep discriminator.
But then we need to see what is the kind of structure which is getting seeped into the reconstruction. Once we do that,
we can then decide upon what should be the best structure for it.

consistent notebook issue:
I observe that when I do not use a batchnormalization, the discriminator loss is quite meaningful.(
/home/ubuntu/ashesh/training/disentangle/2205/D3-M5-S0-L1/4)
Although, I must say that I don't get the training loss which I observe while training. Here
I'm talking about the critic loss. For the reconstruction loss, things match almost perfectly.
However, this does not solve the issue. I see that in the main model, we have used the batch2dnormalization a number of
times and there is no issue there. So, then it does not make sense why
this should affect only the discriminator.

I think before I optimize the critic based setup, I should look into these four experiments as it is very likely that I
may get a better performance by doing them.
Exp 0: Observe how good the performance becomes when increasing the image size just on the validation set.
Exp 1: Enable Learning rate scheduler => rutherford /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/11
Exp 2: Work with 16 bit precision (a baseline for other models)
Exp 3: Image size: Increase the size. (with 16 bit precision)
turing /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/8 256 * 256
tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/13 512 * 512 ( it also has dropout and kl_weights
different)
Performance on 512 sized : Rec:0.006886 Rec L1:0.006736 Rec L2:0.007039 scaled PSNR L1:24.63 PSNR L2:33.22 simple PSNR
L1:23.87 PSNR L2:32.53
Performance on 256 sized : Rec:0.007159 Rec L1:0.007003 Rec L2:0.007334 scaled PSNR L1:21.76 PSNR L2:30.86

Currently, tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/8 is top model trained on 256 * 256.
Its performance on 512 sized: Rec:0.007216 Rec L1:0.007148 Rec L2:0.007345 scaled PSNR L1:24.41 PSNR L2:33.03

Exp 4: Add Rotation in Augmentation on training data (tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/9)
Rec:0.009270 Rec L1:0.009228 Rec L2:0.009337 scaled PSNR L1:20.10 PSNR L2:29.18 simple PSNR L1:19.88 PSNR L2:29.24
#TODO: One thing which I've not done here is that in validation, one needs to get all rotation variants and aggregate
their prediction. One then uses this as the final prediction of the model. =>  done, only slightly improved version.

I think there is one reason as to why this performance might not have improved as much as I would've liked it to.
When you rotate a crop by some angle, you, on average, increase the zero space. One way to avoid it would be to allow
for just 4 rotations. In this case, I'll not have that issue. Another thing is to allow for flipping. I've made those
changes and will shortly start the training.
turing /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/10 (flipping + 90rotation)
I don't see any improvement. Idea now is to find the bottleneck. What is limiting the performance. Atleast the training
error has to go down.
tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/11 (reduced kl_weight=0.005
and increased max_var=8)
tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/12 (increase depth and channel count)
ruth /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/12 (increase depth even further and channel count)
My interpretation of the increased variance is that it allows to capture a larger amount of the subspace. However, the
counter argument is that with larger variance it is difficult to ascertain what would be the sampled z. And so, the
model would essentially behave very similar for nearby mean values. I think, it then makes sense to look at for which
images, do the stdev() in q() gets very high:
a) is it only few worse examples or is it all examples in general.
b) Also, is it for few channels or is it on all channels?

## Exp 0

tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M5-S0-L1/4
128 => Rec:0.010008, Rec L1:0.009271 Rec L2:0.010606, PSNR L1:18.40 PSNR L2:25.37
256 => Rec:0.009056, Rec L1:0.008398 Rec L2:0.009766, PSNR L1:20.29 PSNR L2:28.78
512 => Rec:0.008922, Rec L1:0.008279 Rec L2:0.009643, PSNR L1:23.20 PSNR L2:31.26
512 => Rec:0.008967, Rec L1:0.008279 Rec L2:0.009633, PSNR L1:23.21 PSNR L2:31.26
1024=> Rec:0.009800, Rec L1:0.009087 Rec L2:0.010522, PSNR L1:28.64 PSNR L2:31.26
256 then seems to be the optimal

Exp 5: Look at the optimal PSNR code and use it from there.: I see that below, there is an improvement.

## variant PSNR

tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/9
scaled: PSNR L1:21.24 PSNR L2:30.29
simple: PSNR L1:19.88 PSNR L2:29.24

tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/8
Rec:0.007471 Rec L1:0.007374 Rec L2:0.007601 scaled PSNR L1:21.63 PSNR L2:30.76 simple PSNR L1:20.69 PSNR L2:29.79

tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/10
Rec:0.008336, Rec L1:0.008291 Rec L2:0.008419 scaled PSNR L1:21.48 PSNR L2:30.56 simple PSNR L1:20.47 PSNR L2:29.62

tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/11
Rec:0.007563 Rec L1:0.007373 Rec L2:0.007756 scaled PSNR L1:21.64 PSNR L2:30.76 simple PSNR L1:20.70 PSNR L2:29.86

tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/12
Rec:0.007883 Rec L1:0.007499 Rec L2:0.008265 scaled PSNR L1:21.65 PSNR L2:30.65 simple PSNR L1:20.64 PSNR L2:29.63

8 fold Rotation /8 tur
Rec L1:0.007342 Rec L2:0.007491
PSNR L1:20.71 PSNR L2:29.86

I see that the dropout is also enabled. It could also be the reason why performance on training data has not improved.
Another point is that the performance improvement which I've got could be simply due to reducin the batch size. One
needs to run a baseline to ascertain this.

Exp 6
Enabling the reconstruction loss /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L2/0
Rec:0.007813 Rec L1:0.007780 Rec L2:0.007841 scaled PSNR L1:21.53 PSNR L2:30.81 simple PSNR L1:20.66 PSNR L2:29.88

Exp 7
Reducting the dropout: 0.05 (ruth /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/13)
dropout: 0.0 tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/14
Rec:0.006574
Rec L1:0.006450 Rec L2:0.006737
RMSE L1:0.1103 L2:0.1122
PSNR L1:24.66 PSNR L2:33.33

The variance of P() is fixed to 1. When we give some weight w to the KL divergence term, what this essentially means is
that we are fixing the stdev of P() to w.

## Inspecting what are the images which are performing badly on training data.

## Inspecting what are the images which are performing badly: do rotated images have any inferior performance.

# May 27

## TODO: Zoom-in the image and then compute the PSNR. Zoom-in the image and aggregate reconstructions.

(256 sized)
ruth /home/ubuntu/ashesh/training/disentangle/2205/D3-M5-S0-L1/1: Critic with 0.1 weight
128:
scaled PSNR L1:21.35 PSNR L2:29.98
RMSE L1:0.1172 L2:0.1248
256:
PSNR L1:21.35 PSNR L2:29.98
RMSE L1:0.1172 L2:0.1248

size 128: tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M5-S0-L1/1: Critic with 0.01 weight
size 256: tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M5-S0-L1/6 Critic with 0.005 weight
256:
PSNR L1:21.37 PSNR L2:30.49
RMSE L1:0.1178 L2:0.1195
nucleus + actin
I've changed the channels to be (0,3) /home/ubuntu/ashesh/training/disentangle/2206/D3-M3-S0-L0/0

enabling logvar: /home/ubuntu/ashesh/training/disentangle/2206/D3-M3-S0-L0/0
Idea is to look at the RMSE/scaled PSNR to ascertain whether there is any improvement or not.

chanelwise logvar: tur disentangle/2206/D3-M3-S0-L0/1
RMSE L1:0.1132 L2:0.1154
PSNR L1:24.63 PSNR L2:33.23

global logvar: ruth training/disentangle/2206/D3-M3-S0-L0/1
Rec:-0.724319
Rec L1:-0.727334 Rec L2:-0.719014
RMSE L1:0.1180 L2:0.1190
PSNR L1:24.39 PSNR L2:32.98

pixelwise logvar: ruth training/disentangle/2206/D3-M3-S0-L0/0
Rec:-0.860996 KL:nan
Rec L1:-0.864775 Rec L2:-0.855941
RMSE L1:0.1103 L2:0.1113
PSNR L1:24.71 PSNR L2:33.28

## Why is it the case that the bright blobs in region 2 comes only at boundaries. => I checked the val data loader.

I see all kinds of shapes. However, it could be happening that the bright regions around boundary are the ones with the
best results and so they are more visible.

# June 11

## Comparing with doubledip

In doubledip, we work with 2 weights.
We pick the best 10 and worst 10 image crops from the following trained model and check double dip's performance on it.
tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/14

worst:
Rec:0.008295 KL:nan
Rec L1:0.008302 Rec L2:0.008350
RMSE L1:0.1261 L2:0.1265
PSNR L1:20.34 PSNR L2:30.24

best:
Rec:0.004696 KL:nan
Rec L1:0.004710 Rec L2:0.004724
RMSE L1:0.0969 L2:0.0971
PSNR L1:34.58 PSNR L2:34.51

doubledip (with 2 inputs)
best
PSNR L1:32.75 PSNR L2:29.17
worst
PSNR L1:19.02 PSNR L2:26.96

doubledip (with one input)
best
PSNR L1:26.29 PSNR L2:23.54

## VampPrior idea

The idea is to start simple. We will have just one level VAE. And we will compare it against one level VAE with vamprior
Later, when we extend it to ladder vae, we will have a choice whether to induce vamp prior at each level or
just at the top most level.

Note that here, we take the kl divergence between q and p. We explicitly define q and p distributions and then
use pytorch based code to get this.

However, in the vamprior, for KL divergence, we don't create the P distribution explicitly. we compute kl divergence
using a single sample MC approximation.
I've simplified the setup to start simple with just one resolution level.
ruth /home/ubuntu/ashesh/training/disentangle/2206/D3-M3-S0-L0/6: with vamprior enabled.
Still running.
Validation (512):
Rec:-0.451406 KL:nan
Rec L1:-0.516070 Rec L2:-0.387791
RMSE L1:0.1459 L2:0.1579
PSNR L1:23.22 PSNR L2:30.88

Training (512):
Rec:-0.361301 KL:nan
Rec L1:-0.426261 Rec L2:-0.296286
RMSE L1:0.1605 L2:0.1745
PSNR L1:24.07 PSNR L2:30.73

Validation (256):
Rec:-0.541758 KL:nan
Rec L1:-0.607862 Rec L2:-0.476115
RMSE L1:0.1369 L2:0.1498
PSNR L1:20.73 PSNR L2:28.91

tur /home/ubuntu/ashesh/training/disentangle/2206/D3-M3-S0-L0/3: with vamprior disabled.
Validation(256):
Rec:-0.694106 KL:nan
Rec L1:-0.697464 Rec L2:-0.691742
RMSE L1:0.1190 L2:0.1205
PSNR L1:21.45 PSNR L2:30.39

Training(256):
Rec:-0.650173 KL:nan
Rec L1:-0.654998 Rec L2:-0.645784
RMSE L1:0.1286 L2:0.1303
PSNR L1:22.18 PSNR L2:30.96

Training(512):
Rec:-0.626107 KL:nan
Rec L1:-0.631300 Rec L2:-0.620459
RMSE L1:0.1313 L2:0.1330
PSNR L1:25.17 PSNR L2:32.62

tur /home/ubuntu/ashesh/training/disentangle/2206/D3-M3-S0-L0/4: with vamprior disabled and using the analytical KL
The point of training this is just to check whether using a one sample MC estimate of KL is similar/different to
analytical KL. It is very similar. I don't see any real difference.

1. Check why overall loss is negative. => earlier we were just computing the MSE (logvar was 0). Now, there is also
   logvar in the picture. The logvar is of decoder ~ P(x|z).
2. Check the training performance.
3. Ensure that the notebook reconstruction and training run reconstruction loss matches. => For validation, I'm getting
   the same metrics which I see while training. For training dataset, the numbers are off significantly. I see that the
   training loss is quite consistent across multiple (3) computations without setting the random seeds. I get the same
   value upto 3 decimal places for reconstruction loss.

## TODO: Why the training loss does not match between runs and the notebooks.? => this has to do with model.eval()

If model.eval() is not True, training loss exactly matches between runs and notebooks. So, one idea is to either disable
batchnorm or have more batch size.

## TODO: KL loss is coming out to be negative. One needs to see how that can happen? there must be a bug.

On this I've to say that the shape of the kl divergence curve is quite similar to what I've got when not using the
vampprior. => This was a harmless bug. So, log(sqrt(1/2pi)) was not included while computing the log of the probablity.

Another issue is that for Vampprior, the KL divergence is getting negative. which is slightly disturbing.
Another issue is that in the code, we do clipping of the logvar of q_params and p_params. however, we still go on using
the p_params and q_params.
Ideally, we should not use q_params and p_params, but instead work with the mu and the sigma as it is clipped. => fixed
it.

tur /home/ubuntu/ashesh/training/disentangle/2206/D3-M3-S0-L0/5: just one input for the vampprior. validation is super
fluctuating and has high error.

tur /home/ubuntu/ashesh/training/disentangle/2206/D3-M3-S0-L0/12: with 20 inputs for the vamprior
Validation(256):
Rec:-0.545016 KL:-0.832505
Rec L1:-0.586590 Rec L2:-0.504542
RMSE L1:0.1351 L2:0.1482
PSNR L1:20.88 PSNR L2:28.64

Training(256):
Rec:-0.459280 KL:-0.831866
Rec L1:-0.496504 Rec L2:-0.422399
RMSE L1:0.1506 L2:0.1635
PSNR L1:21.58 PSNR L2:29.02

I see that it is quite similar to ruth /home/ubuntu/ashesh/training/disentangle/2206/D3-M3-S0-L0/6. In other words
adding more number of inputs has not helped. (yet)
But the thing is that in the code, they had used 500 as number of inputs. Also, ideally, the number of inputs should
be the entire training dataset. So, it makes sense to experiment with larger number of trainable inputs.
So, to achieve this I'm reducing the image size by a factor of 4 to 64. This allows me to have upto 300 trainable
inputs.

ruth /home/ubuntu/ashesh/training/disentangle/2206/D3-M3-S0-L0/8: vamprior disabled. (image_size:64) std has very high
values
and it crashed. running it with lower max threshold.
ruth /home/ubuntu/ashesh/training/disentangle/2206/D3-M3-S0-L0/9: with max_log_var set to 6. So, I see that it is not
related to max_log_var. The self.top_prior_params is set to Nan. The reason is that the KL loss gets super high. q_var
is also touching maximum value 6. Note that when the input size was 256, it was not the case. it was also fluctuating
but did not get to nan. When I compare the analytical KL and one example estimate, I don't see any clear difference
between them at 256*256 resolution (/3 and /4)
tur /home/ubuntu/ashesh/training/disentangle/2206/D3-M3-S0-L0/17: vamprior enabled. (image_size:64)
Another thing to look is that how is the KL divergence loss behaves on the validation set.

Now, I've adopted a two optimizer based setup. The motivations are two:

1. Any non-ELBO term is historically seen to disturb the data extraction process of the decoder. I read is somewhere, in
   some paper. (some? :D :( )
2. I don't want to allow the model to change the decoder weights so that they align with the trainable input well to
   together yield low KL divergence. Doing this should encourage the learning of learnable inputs of the vampprior

Without dropout:ruth /home/ubuntu/ashesh/training/disentangle/2206/D3-M6-S0-L0/0
Rec:-0.482274 KL:nan
Rec L1:-0.536763 Rec L2:-0.427336
RMSE L1:0.1432 L2:0.1607
PSNR L1:20.79 PSNR L2:28.16
The validation loss is quite bumpy and training loss is quite smooth.

Adding dropout: tur /home/ubuntu/ashesh/training/disentangle/2206/D3-M6-S0-L0/1
Rec:-0.624432 KL:nan
Rec L1:-0.661199 Rec L2:-0.588311
RMSE L1:0.1298 L2:0.1379
PSNR L1:21.00 PSNR L2:29.46
The validation is relatively smooth.

Since with dropout, things started working better, the question would be that how is the performance
with dropout on single optimizer case.
rutn /home/ubuntu/ashesh/training/disentangle/2206/D3-M3-S0-L0/12

and with dropout but without vampprior
tur /home/ubuntu/ashesh/training/disentangle/2206/D3-M3-S0-L0/20

I see that dropout + (no vampprior) is giving the best performance. Which essentially means that
vampprior is not working.

tur /home/ubuntu/ashesh/training/disentangle/2206/D3-M3-S0-L0/21
Here, I'm working with a non-optimized version of vampprior. Just want to check if this one also gives similar
performances.

## Another idea is that make it a proper VAE. Just keep the penultimate layer having channels 2 and enforce a loss on one of them and the loss on the final output.

It will be a VAE with one additional constraint on the penultimate layer.

Since I know that the batch normalization is probably the reason why training performance becomes so very different
in notebook depending upon whether I do model.eval() or not, it might make sense to train a model without batch
normalization.

## Enabling contrastive learning.

ruth /home/ubuntu/ashesh/training/disentangle/2206/D3-M3-S0-L0/13: existing setup (without CL) with randomized data
loader.
After looking at the performance of above, I think that it may not be the best option to induce contrastive learning
here. The reason is that for me to enable contrastive learning, I need to decouple the two channels. So, I cannot take
co-occuring channels. But since the performance of randomized data loader is poor, this means that decoupling has an
inferior affect on the validation data and so there is no reason to support that CL will help improve the validation
performance. Although I coded up the two index data loader and sampler, I'm not thinking of going forward with it.

However, another interpretation is also there. Now that the two channels are decoupled using same decoder does not make
sense anymore. It would be better to have two decoders which do this job. The hope is that they would be able to extract
a better performance out. This is also evident from the inferior training performance of the this model over the case
when two channels were co-localized.
tur /home/ubuntu/ashesh/training/disentangle/2206/D3-M4-S0-L0/0: this has twin decoder setup. This is used with non
co-occuring channels. The expectation now is that we would be able to get better reconstruction loss on training data
than ruth /home/ubuntu/ashesh/training/disentangle/2206/D3-M3-S0-L0/13. The hope is that we get better validation
performance than the above mentioned model and may be with the case when it is absent. On a different note, this is
equivalent to training a single VAE with just one target channel. We can anyway learn the other channel by subtracting
from the input. There is one condition where I think using two channels in output will be better over using one channel.
Lets say that the model is confident about one channel on a speciic region. But the model does not know what should be
the channel 2. In that case, the model iwll disturb the prediction of channel 1 as well. but if the counter argument is
thati if one of the channel is super clear, there is no reason to beleivethat the modeldoes not have this sense of
subtracting that from the input. This is slightly hazy.

another thing is to check how the performance changes when one does not do model.eval()
/home/ubuntu/ashesh/training/disentangle/2206/D3-M3-S0-L0/12
With eval():
Rec:-0.548259 KL:nan
Rec L1:-0.619266 Rec L2:-0.476884
RMSE L1:0.1324 L2:0.1540
PSNR L1:20.86 PSNR L2:28.42

Without eval():
Rec:-0.543836 KL:nan
Rec L1:-0.645605 Rec L2:-0.441391
RMSE L1:0.1334 L2:0.1546
PSNR L1:21.13 PSNR L2:29.51

another thing is to inspect the performance with and without the rotation. I remember doing it once. I think it was
understood that the model does not give inferior performance to rotated versions as well. But this is not completely
clear.

TODO: look at the distribution of the latent space