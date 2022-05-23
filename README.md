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
/home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/9: stochastic_skip=False

With MSE, I see that the best cases are mainly those where there is very little content. In that sense PSNR might be a
better metric to observe things.

To fix the variance exploding problem I've added a maximum value limit on the variance. I simply use clipping.
/home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/3

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
/home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/10)
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
Exp 3: Image size: Increase the size. (with 16 bit precision) turing
/home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/8
Exp 4: Add Rotation in Augmentation on training data (tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/9)
Rec:0.009270 KL:nan
Rec L1:0.009228 Rec L2:0.009337
PSNR L1:20.10 PSNR L2:29.18
#TODO: One thing which I've not done here is that in validation, one needs to get all rotation variants and aggregate
their prediction. One then uses this as the final prediction of the model.

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

Exp 5: Look at the optimal PSNR code and use it from there.

## Exp 0

tur /home/ubuntu/ashesh/training/disentangle/2205/D3-M5-S0-L1/4
128 => Rec:0.010008, Rec L1:0.009271 Rec L2:0.010606, PSNR L1:18.40 PSNR L2:25.37
256 => Rec:0.009056, Rec L1:0.008398 Rec L2:0.009766, PSNR L1:20.29 PSNR L2:28.78
512 => Rec:0.008922, Rec L1:0.008279 Rec L2:0.009643, PSNR L1:23.20 PSNR L2:31.26
512 => Rec:0.008967, Rec L1:0.008279 Rec L2:0.009633, PSNR L1:23.21 PSNR L2:31.26
1024=> Rec:0.009800, Rec L1:0.009087 Rec L2:0.010522, PSNR L1:28.64 PSNR L2:31.26
256 then seems to be the optimal

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
Rec:0.210035 KL:nan
Rec L1:0.007342 Rec L2:0.007491
PSNR L1:20.71 PSNR L2:29.86
It appears that the rotated versions give very high errors. One would need to see individually how they perform. here,
Rec:0.21 is the reconstruction loss on a rotated version of the input. it is way high

I see that the dropout is also enabled. It could also be the reason why performance on training data has not improved.
Another point is that the performance improvedment which I've got could be simply due to reducin the batch size. One
needs to run a baseline to ascertain this.

## Inspecting what are the images which are performing badly on training data.

## Inspecting what are the images which are performing badly: do rotated images have any inferior performance.