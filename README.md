# Objective
Here, our goal is to disentangle the images by taking as input the sum of the two images. For example, given image A & B, input to our model is A+B and the output is A,B

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

Varying (model.z_dims, Num of filters)
    ruth:   (32,16)   Rec:0.0153 KL:0.0030      /home/ubuntu/ashesh/training/disentangle/2203/D1-M3-S1-L0/1
    turing: (128, 64) Rec:0.0104 KL:0.0027      /home/ubuntu/ashesh/training/disentangle/2203/D1-M3-S1-L0/1
    turing: (256,128) Rec:0.0091 KL:0.0021      /home/ubuntu/ashesh/training/disentangle/2203/D1-M3-S1-L0/2

Varying kl_weight
Server      kl_weight   Metrics                                 Ckpt
    tur:    1           Rec:0.0104 KL:0.0027    /home/ubuntu/ashesh/training/disentangle/2203/D1-M3-S1-L0/1
    ruth:   0.5         Rec:0.0099 KL:0.0044    /home/ubuntu/ashesh/training/disentangle/2203/D1-M3-S1-L0/2
    tur:    0.2         Rec:0.0089 KL:0.0076    /home/ubuntu/ashesh/training/disentangle/2203/D1-M3-S1-L0/3
    ruth:   0.1         Rec:0.0087 KL:0.0115    /home/ubuntu/ashesh/training/disentangle/2203/D1-M3-S1-L0/3
    tur:    0.05        Rec:0.0088 KL:0.0174    /home/ubuntu/ashesh/training/disentangle/2203/D1-M3-S1-L0/4
    tur:    0.01        Rec:0.0082 KL:0.0332    /home/ubuntu/ashesh/training/disentangle/2203/D1-M3-S1-L0/5

I observe that the validation is not robust enough as the number of samples are less.
One can easily fix this by increasing the random pairs.  