# Audio-separator-based-on-Neural-Network
The MATLAB reproduce of Self-supervised learning Audio separator in paper "Deep Transform: Cocktail Party Source Separation via Complex Convolution in a Deep Neural Network "
## separation method
The input of the neural network is the amplitude spectrum of the mixed (male + female) audio.The goal is an ideal soft mask for male speakers.The loss function is to minimize the mean square error between its output and input targets.At the output, the audio STFT is converted back to the time domain using the output amplitude spectrum and the phase of the mixing signal, using a self-supervised learning method.The training set is the result of separating the ideal soft mask and binary mask for the first 40 seconds of mixed audio, and the network output is the ideal soft mask for predicting male speakers 20 seconds after verifying the set is mixed audio。
## Separation results
