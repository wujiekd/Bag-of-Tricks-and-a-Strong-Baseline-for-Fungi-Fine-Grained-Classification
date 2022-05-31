##finetune
# python train.py /home/data1/lkd/Fungi_data/Fungidata/DF20 -c configs/swin_large_384.yaml \
#         --output /home/data1/lkd/Fungi_data/output/Swin-TF/DF20/All_aug/freeze_layer_2 \
#         --initial-checkpoint /home/data1/lkd/Fungi_data/output/Swin-TF/DF20/freeze_layer_2/20220430-123449-swin_large_patch4_window12_384-384/Best_Top1-ACC.pth.tar \
#         --freeze-layer 2 \
#         --lr 0.001 \
#         --batch-size 32 \
#         --warmup-epochs 0 \
#         --cutmix 1 \
#         --color-jitter 0.4 \
#         --reprob 0.25 \
#         --aa trivial \
#         --decay-rate 0.1


# 训练Swin-transformer 冻结2
# python train.py /home/data1/lkd/Fungi_data/Fungidata/DF20 -c configs/swin_large_384.yaml \
#         --freeze-layer 2 \
#         --batch-size 32 \
#         --lr 0.01 \
#         --output /home/data1/lkd/Fungi_data/output/Swin-TF/DF20/freeze_layer_2 \



# python train_all.py /home/data1/lkd/Fungi_data/Fungidata/DF20 -c configs/swin_large_384.yaml \
#          --batch-size 32 \
#          --img-size 384 \
#          --output /home/data1/lkd/Fungi_data/output/Swin-TF/DF20/All_data/swin_large_384 \
#          --freeze-layer 2 \
#          --initial-checkpoint /home/data1/lkd/Fungi_data/output/Swin-TF/DF20/All_aug/freeze_layer_2/20220502-114033-swin_large_patch4_window12_384-384/Best_Top1-ACC.pth.tar \
#          --lr 0.001 \
#          --cutmix 1 \
#          --color-jitter 0.4 \
#          --reprob 0.25 \
#          --aa trivial \
#          --decay-rate 0.1 \
#          --warmup-epochs 0 \
#          --epochs 24 \
#          --sched multistep \
#          --checkpoint-hist 24

# python Cam_train.py /home/data1/lkd/Fungi_data/Fungidata/DF20 -c configs/efficient_b6.yaml \
#          --batch-size 16 \
#          --img-size 600 \
#          --output /home/data1/lkd/Fungi_data/output/Cam/efficient_b6 \
#          --freeze-layer 1 \
#          --initial-checkpoint /home/data1/lkd/Fungi_data/output/All_data/eff_b6/checkpoint-20.pth.tar \
#          --lr 0.0001 \
#          --color-jitter 0.4 \
#          --decay-rate 0.1 \
#          --warmup-epochs 0 \
#          --epochs 18 \
#          --sched multistep \
#          --checkpoint-hist 18



#python Cam_train.py /home/data1/lkd/Fungi_data/Fungidata/DF20 -c configs/efficient.yaml \
#          --batch-size 16 \
#          --img-size 600 \
#          --output /home/data1/lkd/Fungi_data/output/Cam/efficient_b7 \
#          --freeze-layer 1 \
#          --initial-checkpoint /home/data1/lkd/Fungi_data/output/All_data/eff_b7/checkpoint-20.pth.tar \
#          --lr 0.0001 \
#          --color-jitter 0.4 \
#          --decay-rate 0.1 \
#          --warmup-epochs 0 \
#          --epochs 18 \
#          --sched multistep \
#          --checkpoint-hist 18


## 多卡训练tf_efficientnet_l2_ns
#python -m torch.distributed.launch --nproc_per_node=4 train.py /home/data1/lkd/Fungi_data/Fungidata/DF20 -c configs/tf_efficientnet_l2_ns.yaml \
#         --freeze-layer 2 \
#         --batch-size 4 \
#         --lr 0.01 \
#         --output /home/lkd22/fungi/Fungi_data/output/tf_efficientnet_l2_ns/freeze_layer_2 \
#         --sync-bn \
#         --cutmix 1 \
#         --color-jitter 0.4 \
#         --reprob 0.25 \
#         --aa trivial \
#         --decay-rate 0.5

python -m torch.distributed.launch --nproc_per_node=4 train_all.py /home/data1/lkd/Fungi_data/Fungidata/DF20 -c configs/efficient_b6.yaml \
         --freeze-layer 6 \
         --batch-size 8 \
         --lr 0.001 \
         --output /home/lkd22/fungi/Fungi_data/output/eff_b6/freeze_layer_6/balance \
         --initial-checkpoint /home/data1/lkd/Fungi_data/Fungidata/linshi_model/eff_b6_last/checkpoint-20.pth.tar  \
         --sync-bn \
         --color-jitter 0.4 \
         --reprob 0.25 \
         --decay-rate 0.1 \
         --aa trivial \
         --epochs 5 \
         --warmup-epochs 0