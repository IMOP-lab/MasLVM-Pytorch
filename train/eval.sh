python -m torch.distributed.launch --nproc_per_node=1 maslvm.py \
    --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth \
    --model-type vit_l \
    --output work_dirs/maslvm \
    --eval \
    --world_size 1 \
    --batch_size_train 1 \
    --batch_size_valid 1 \
    --restore-fft ./pretrained_checkpoint/fft_epoch_19.pth \
    --restore-bod ./pretrained_checkpoint/bod_epoch_19.pth \
    --restore-model ./pretrained_checkpoint/epoch_19.pth \

