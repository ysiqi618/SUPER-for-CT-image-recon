# SUPER-for-CT-recon
## Install MIRT toolbox ############
get the IRT https://web.eecs.umich.edu/~fessler/code/ unzip in to ./irt (the root folder of this project).
## Install MatConvNet #############
http://www.vlfeat.org/matconvnet/install/#compiling and http://www.vlfeat.org/matconvnet/install/, install to the root folder of this project.
## trained models 
Trained models are included in the root folder "trained_model".

## Code: SUPER with WavResNet #############
folder "super_wavresnet":
1. SUPER-WRN-EP:
- train: main_train_wrsEP_l2normReg_6pat_nufft.m
- test: test_mytrain_mayo_wavEP_l2normReg_nufft.m
2. SUPER-WRN-ULTRA:
- train: main_train_wrsULTRA_6pat_nufft.m
- test: test_mytrain_mayo_wavULTRA_l2normReg_nufft.m
3. standalone WavResNet (based on [1]) and Sequential Supervised Networks:
- train: main_train_wrs_6pat_nufft.m
- test: test_mytrain_rnn_mayo_nufft.m 


## Code: SUPER with FBPConvNet ##########
folder "super_fbpconvnet":
1. SUPER-FCN-EP:
- train: main_train_fbpconvnetep.m
- test:  evaluation_fbpconvnetep.m
2. SUPER-FCN-ULTRA:
- train: main_train_fbpconvnetultra.m
- test:  evaluation_fbpconvnetultra.m
3. Sequential fbpconvnet:
- train: main_train_successivefcn.m
- test:  evaluation_successivefcn.m
4. SUPER-FCN-DataTermOnly:
- train: main_train_fbpconvnetnoreg.m
- test:  evaluation_fbpconvnet_dataterm.m
5. standalone FBPConvNet (based on [2]):
- train: main_train_fbpconvnet.m
- test:  evaluation_fbpconvnet.m

## Code: Standalone Unsupervised Methods (based on [3]):
1. PWLS-EP: main_fan_pwls_ep_nufft.m
2. PWLS-ULTRA: main_fan_pwls_ultra_nufft.m

## Citation
@article{SUPER:20:TMI, \
  title={Unified Supervised-Unsupervised (SUPER) Learning for X-ray CT Image Reconstruction}, \
  author={S. Ye and Z. Li and M. T. McCann and Y. Long and S. Ravishankar},\
  journal={arXiv preprint arXiv:2010.02761},
  year={2020}
}


## References
1. https://github.com/jongcye/deeplearningLDCT
2. https://github.com/panakino/FBPConvNet
3. https://github.com/xuehangzheng/PWLS-ULTRA-for-Low-Dose-3D-CT-Image-Reconstruction


If you have any problems in using it, please contact 
yesiqi@sjtu.edu.cn and zhipengli@sjtu.edu.cn
 
