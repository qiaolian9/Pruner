
# ===========   online model: siamese transfer learning strategy    ==============
# sh tune.sh resnet_50 2000 pam-siamese-update asplos 0 a100_40 pam_k80_1500
# sh tune.sh deeplabv3_resnet50 2000 pam-siamese-update asplos 1 a100_40 pam_k80_1500
# sh tune.sh mobilenet_v2 2000 pam-siamese-update asplos 1 a100_40 pam_k80_1500
# sh tune.sh inception_v3 2000 pam-siamese-update asplos 1 a100_40 pam_k80_1500
# sh tune.sh wide_resnet_50 2000 pam-siamese-update asplos 1 a100_40 pam_k80_1500
# sh tune.sh detr 2000 pam-siamese-update asplos 1 a100_40 pam_k80_1500
# sh tune.sh vit 2000 pam-siamese-update asplos 1 a100_40 pam_k80_1500
# sh tune.sh densenet_121 2000 pam-siamese-update asplos 1 a100_40 pam_k80_1500
# sh tune.sh bert_base 2000 pam-siamese-update asplos 1 a100_40 pam_k80_1500
# sh tune.sh bert_tiny 2000 pam-siamese-update asplos 1 a100_40 pam_k80_1500


# ===========    online model: normal update strategy    ==============
# sh tune.sh resnet_50 2000 pam asplos 0 a100_40 
# sh tune.sh deeplabv3_resnet50 2000 pam asplos 1 a100_40 
# sh tune.sh mobilenet_v2 2000 pam asplos 1 a100_40 
# sh tune.sh inception_v3 2000 pam asplos 1 a100_40 
# sh tune.sh wide_resnet_50 2000 pam asplos 1 a100_40 
# sh tune.sh detr 2000 pam asplos 1 a100_40 
# sh tune.sh vit 2000 pam asplos 1 a100_40 
# sh tune.sh densenet_121 2000 pam asplos 1 a100_40 
# sh tune.sh bert_base 2000 pam asplos 1 a100_40 
# sh tune.sh bert_tiny 2000 pam asplos 1 a100_40 


# ===========    offline cost model: fine-tune on target data    ==============
sh tune.sh resnet_50 2000 pam-no-update asplos 0 a100_40 fine_tune_pam_a100
# sh tune.sh deeplabv3_resnet50 2000 pam-no-update asplos 1 a100_40 fine_tune_pam_a100_361_361
# sh tune.sh mobilenet_v2 2000 pam-no-update asplos 1 a100_40 fine_tune_pam_a100_361_361
# sh tune.sh inception_v3 2000 pam-no-update asplos 1 a100_40 fine_tune_pam_a100_361_361
# sh tune.sh wide_resnet_50 2000 pam-no-update asplos 1 a100_40 fine_tune_pam_a100_361_361
# sh tune.sh detr 2000 pam-no-update asplos 1 a100_40 fine_tune_pam_a100_361_361
# sh tune.sh vit 2000 pam-no-update asplos 1 a100_40 fine_tune_pam_a100_361_361
# sh tune.sh densenet_121 2000 pam-no-update asplos 1 a100_40 fine_tune_pam_a100_361_361
# sh tune.sh bert_base 2000 pam-no-update asplos 1 a100_40 fine_tune_pam_a100_361_361
# sh tune.sh bert_tiny 2000 pam-no-update asplos 1 a100_40 fine_tune_pam_a100_361_361
