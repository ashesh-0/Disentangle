# Objective
Here, our goal is to disentangle the images by taking as input the sum of the two images. For example, given image A & B, input to our model is A+B and the output is A,B

Example run:
python disentangle/scripts/run.py --workdir=/home/ubuntu/ashesh/training/disentangle/ -mode=train --datadir=/mnt/ashesh/places365_noisy/ --config=/home/ubuntu/ashesh/code/Disentangle/disentangle/configs/places_lvae_twindecoder_config.py

## Exp 1
Tried two samplers: Random sampler and SingleImg sampler. In first, a batch comprises of two random samples. (/home/ubuntu/ashesh/training/disentangle/2202/D2-M3-S2-L0/0)
In second a batch comprises of 1 image and other random samples. (/home/ubuntu/ashesh/training/disentangle/2202/D2-M3-S2-L0/1)
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

Varying (model.z_dims,  Val Metrics                 Train Metrics
        Num of filters)
    ruth:   (32,16)     Rec:0.0153 KL:0.0030      Rec:0.014404 KL:0.002947      disentangle/2203/D1-M3-S1-L0/1
    turing: (128, 64)   Rec:0.0104 KL:0.0027      Rec:0.007735 KL:0.002670      disentangle/2203/D1-M3-S1-L0/1
    turing: (256,128)   Rec:0.0091 KL:0.0021      Rec:0.005894 KL:0.002202      disentangle/2203/D1-M3-S1-L0/2

Varying kl_weight
Server      kl_weight   Val Metrics             Train Metrics                                 Ckpt
    tur:    1           Rec:0.0104 KL:0.0027    Rec:0.007735 KL:0.002670    disentangle/2203/D1-M3-S1-L0/1
    ruth:   0.5         Rec:0.0099 KL:0.0044    Rec:0.007674 KL:0.004390    disentangle/2203/D1-M3-S1-L0/2
    tur:    0.2         Rec:0.0089 KL:0.0076    Rec:0.005330 KL:0.007597    disentangle/2203/D1-M3-S1-L0/3
    ruth:   0.1         Rec:0.0087 KL:0.0115    Rec:0.004557 KL:0.011589    disentangle/2203/D1-M3-S1-L0/3
    tur:    0.05        Rec:0.0088 KL:0.0174    Rec:0.006042 KL:0.017418    disentangle/2203/D1-M3-S1-L0/4
    tur:    0.01        Rec:0.0082 KL:0.0332    Rec:0.003737 KL:0.033354    disentangle/2203/D1-M3-S1-L0/5

I observe that the validation is not robust enough as the number of samples are less.
One can easily fix this by increasing the random pairs.  I checked it in validation. it is not much significant.

blocks per layer =5
tur:                    Rec:0.007762 KL:0.016818    Rec:0.005100 KL:0.017000 disentangle/2204/D1-M3-S1-L0/0



z_dims from 3 to 2
    ruth:   2        Rec:0.0087 KL:0.0209    Rec:0.0064 KL:0.0206    disentangle/2204/D1-M3-S1-L0/0
    tur:    3        Rec:0.0088 KL:0.0174    Rec:0.0060 KL:0.0174    disentangle/2203/D1-M3-S1-L0/4


I see strong overfitting when kl_weight=0.01. Adding dropout to see if it improves performance.
Server      dropout      Val Metrics             Train Metrics                                 Ckpt
    ruth:   0.2         Rec:0.0081 KL:0.0400    Rec:0.0054 KL:0.0399    disentangle/2204/D1-M3-S1-L0/1
    tur:    0.0         Rec:0.0082 KL:0.0332    Rec:0.0037 KL:0.0334    disentangle/2203/D1-M3-S1-L0/5


I checked that KL divergence calculation is fine in the code.
Twindecoder:    
Server      Config          Val Metrics
tur     kl_weight: 0.1      Rec:0.008110 KL:0.007804    Rec:0.006085 KL:0.007914    disentangle/2204/D1-M4-S1-L0/0
ruth    kl_weight: 0.01     Rec:0.007787 KL:0.026159    Rec:0.005435 KL:0.026358    disentangle/2204/D1-M4-S1-L0/0
In order to overfit most info, I've reduced the kl_weight to 0.0001
ruth /home/ubuntu/ashesh/training/disentangle/2204/D1-M4-S1-L0/1


# Working with microscopy data
Now, I'm working with OptiMEM100x014 dataset. It has total of 61 images.
I see that val loss shoots to infinity? 
Trying a 0.995  quantile as the higher threshold. I see that both image types have same threshold value.

There is asymmetry in the losses of the two classes. one class has more data. other class has less data.
So, the loss is mostly dominated by the one class.

/home/ubuntu/ashesh/training/disentangle/2204/D3-M3-S0-L0/2 is the first working model. 
Notes:
    It estimates one class better than other. Better performing class typically has more pixels with a pattern.
    To fix this, I've code up a hack. Here, one pair of images is valid only if both of them have content.
    In the previous model, if one of them had a content, then we considered it a valid pair. That approach is more real world aligned. However, for the heck of trying it out, I'm trying it to see whether I get better performance
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

I think one may repeat the training factor to 10 times as well. This will help in ensuring that most time is not spent on running the validation only.
Increased the batch_size to 8 and introduced the training factor.
/home/ubuntu/ashesh/training/disentangle/2204/D3-M3-S0-L0/8

To ascertain that the discripency in the validation error is just in due to the fact that we have too few of images and random sampling of patches leads to the discripency, I've created a deterministic data loader. It samples all the grid patches. So the validation error should be deterministic now.
/home/ubuntu/ashesh/training/disentangle/2204/D3-M3-S0-L0/9

I found out the bug. It was that the mean() and std() on training data was different than on validation data.
In the notebook I was using the validation data's mean and std due to which this was happening. Now, the validation performance while training and the performance in notebook matches very closely.


I tried to sample a reconstruction multiple times and then take the average.  However, it has no real benefit as the loss decreases only by 10e-4 units. If I increase the KL weight, then it may become more significant.

/home/ubuntu/ashesh/training/disentangle/
config                                          with deterministic      randomized
2204/D3-M3-S0-L0/9      trained with determ                              0.010+-0.001
2204/D3-M3-S0-L0/7      trained with random                              0.020+-0.002
2204/D3-M3-S0-L0/10     trained with random

## May 7
I see that the mean and the std of the two channels are different. It therefore makes sense to have different mean and std for normalization.

Using different mean and stdev for the two channels.: 
/home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/1
I see that it is not giving a good performance. If this is just because the separate mean std doesn't work, 
then passing the original value of the mean and std should lead to identical performance. Working for that.

same mean and std: /home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/2
todo: skip the mean/std computtaion everytime => DONE
VAL loss seems to be quite fluctuating. It could be so because of either of the two issues:
1. The threshold is inappropriately high and so training is getting slightly skewed. To test this, I probably just need to reduce the threshold to 0 and check. 

2. Everytime one gets a different validation set due to random sampling and so we observe very different results. Since we are using quite high repeat_factor, I don't think that would be the case. To check if this is the case, I could simply evaluate it to see how different are the results that I'm getting.
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

I've figured out what the issue was. When normalizing in the data loader, I need to always use the mean and std of training data and not the validation data. Currently, I was using mean and std of validation data for validation data loader and mean and std of training data. 

/home/ubuntu/ashesh/training/disentangle/2205/D3-M3-S0-L0/4