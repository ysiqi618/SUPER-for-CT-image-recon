# SUPER-for-CT-recon
## ############# Install MIRT toolbox ############
get the IRT https://web.eecs.umich.edu/~fessler/code/ unzip in to ./irt (the root folder of this project).
## ############ Install MatConvNet #############
http://www.vlfeat.org/matconvnet/install/#compiling and http://www.vlfeat.org/matconvnet/install/, install to the root folder of this project.
## trained models 
Trained models are included in the root folder "trained_model".
## ###### SUPER with WavResNet #############
folder "super_wavresnet":
1. SUPER-WRN-EP:
- train: main_train_wrsEP_l2normReg_6pat_nufft.m
- test: test_mytrain_mayo_wavEP_l2normReg_nufft.m
2. SUPER-WRN-ULTRA:
- train: main_train_wrsULTRA_6pat_nufft.m
- test: test_mytrain_mayo_wavULTRA_l2normReg_nufft.m
3. standalone WavResNet and Sequential Supervised Networks
- train: main_train_wrs_6pat_nufft.m
- test: test_mytrain_rnn_mayo_nufft.m 


## ########### SUPER with FBPConvNet ##########



 
