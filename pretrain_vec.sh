#conda init 2022project

#python main.py --batch_size 64 --epochs 200 --output_dir weights/data2vec_cls_only/ --enc_layer 2 --dec_layers 2 --device cuda:3 --frame_skip 1 --stack_avg 64 --hidden_dim 256 --nheads 8 #--model_debug #--eval

for var in 8 #72 #105 #44 71 45 72 #44 45 #61 62 63 64 84 #65 #105 37 39 84 46 #61 62 63 64 untrained? #105
do 
    #python main.py --batch_size 128 --epochs 90 --output_dir weights/pretrain_uai/vec_final/mask/75/block/$var --in_chan 16 --enc_layers 4 --dec_layers 2 --device cuda:0 --frame_skip 1 --stack_avg 64 --nheads 4 --slow_time 1 --model_scale 't' --kernel 8 --hidden_dim 128 --stride 4 --bbox_loss_coef 0 --mask_ratio 0.75 --pretrain --lr 4e-5 --lr_min 1e-5 --raw_loss_coef 0 --weight_decay 5e-2 --inv_mask_block $var #--model_debug #--attn_mean #--model_debug --attn_mean #--model_debug #--posemodel  #--model_debug #--resume weights/scratch_convnext16token_downsamplex_vitx/checkpoint.pth #--model_debug
    python main.py --batch_size 128 --epochs 90 --output_dir weights/pretrain_uai/vec_review/channelwise/mask75/block/$var --in_chan 16 --enc_layers 4 --dec_layers 2 --device cuda:0 --frame_skip 1 --stack_avg 64 --nheads 4 --slow_time 1 --model_scale 't' --kernel 8 --hidden_dim 128 --stride 4 --bbox_loss_coef 0 --mask_ratio 0.75 --pretrain --lr 4e-5 --lr_min 1e-5 --raw_loss_coef 0 --weight_decay 5e-2 --inv_mask_block $var --embtype 'channel_wise' #--model_debug #--attn_mean #--model_debug --attn_mean #--model_debug #--posemodel  #--model_debug #--resume weights/scratch_convnext16token_downsamplex_vitx/checkpoint.pth #--model_debug
done


    
#done
