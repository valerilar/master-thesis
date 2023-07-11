This work uses the Microsoft Scalable Noisy Speech Dataset (MS-SNSD), which can be found at the link https://github.com/microsoft/MS-SNSD

Network 1. This network has an encoder-decoder structure and combines convolutional and LSTM layers. 

Network 2. This network has a U-Net structure and converts noisy signals into spectrograms using STFT. Denoised spectrograms are converted using ISTFT to obtain denoised signals. This network requires additional experiments.

Network 3. Is a combination of three neural networks and is under development.

To run the code, you need to install all the necessary libraries and generate a dataset that will contain the noisy language and the corresponding clean samples. You also need to specify the correct path to your files.
