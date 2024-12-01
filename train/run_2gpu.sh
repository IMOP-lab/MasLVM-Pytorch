export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 maslvm.py \
    --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth \
    --model-type vit_l \
    --output work_dirs/output \
    --batch_size_train 1 \
    --max_epoch_num 20 \
    --distributed \
    --learning_rate 0.001 \



