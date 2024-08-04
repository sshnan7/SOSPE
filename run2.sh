#conda init 2022project


#done

for var in 0 1 2 3 #72 #105 #44 71 45 72 #44 45 #61 62 63 64 84 #65 #105 37 39 84 46 #61 62 63 64 untrained? #105
do
    python main.py --batch_size 32 --epochs 25 --output_dir weights/CIKM_2024/one_subgroup/$var --in_chan 16 --enc_layers 4 --dec_layers 2 --device cuda:0 --frame_skip 1 --stack_avg 64 --nheads 4 --group_idx $var --slow_time 1 --model_scale 't' --kernel 8 --hidden_dim 128 --stride 4 --bbox_loss_coef 0 --posemodel --lr 4e-4 --lr_min 4e-5 --raw_loss_coef 0 #--embtype 'channel_wise' #--model_debug #--attn_mean #--model_debug --attn_mean #--model_debug #--posemodel  #--model_debug #--resume weights/scratch_convnext16token_downsamplex_vitx/checkpoint.pth #--model_debug
done
