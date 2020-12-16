# ldct_nonlocal
Official Source Code for the paper Noise Conscious Training of Non Local Neural Network Powered by Self Attentive Spectral Normalized Markovian Patch GAN for Low Dose CT Denoising.

# Dependencies:
Pytorch

Pydicom

## DATASET

The 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge by Mayo Clinic   
https://www.aapm.org/GrandChallenge/LowDoseCT/

The `data_path` should look like:


    data_path
    ├── L067
    │   ├── quarter_3mm
    │   │       ├── L067_QD_3_1.CT.0004.0001 ~ .IMA
    │   │       ├── L067_QD_3_1.CT.0004.0002 ~ .IMA
    │   │       └── ...
    │   └── full_3mm
    │           ├── L067_FD_3_1.CT.0004.0001 ~ .IMA
    │           ├── L067_FD_3_1.CT.0004.0002 ~ .IMA
    │           └── ...
    ├── L096
    │   ├── quarter_3mm
    │   │       └── ...
    │   └── full_3mm
    │           └── ...      
    ...
    │
    └── L506
        ├── quarter_3mm
        │       └── ...
        └── full_3mm
                └── ...     
## USE
use the prep.py file to generate the traning and test data.

update the data path in train_model.py and test_model.py

run the file train_model.py to train

## Acknowledgments
* Our code architecture is inspired by [JiahuiYu](https://github.com/JiahuiYu/generative_inpainting) and [SSinyu](https://github.com/SSinyu/RED-CNN). 
 
