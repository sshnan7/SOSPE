#conda init 2022project

#python main.py --batch_size 64 --epochs 200 --output_dir weights/data2vec_cls_only/ --enc_layer 2 --dec_layers 2 --device cuda:3 --frame_skip 1 --stack_avg 64 --hidden_dim 256 --nheads 8 #--model_debug #--eval

#python main.py --batch_size 64 --epochs 200 --output_dir weights/data2vec_final_mse_only/ --enc_layer 2 --dec_layers 2 --device cuda:0 --frame_skip 1 --stack_avg 64 --hidden_dim 256 --nheads 8 #--model_debug #--eval


#for var in 85
for var in 0
do
    #python main.py --batch_size 64 --epochs 25 --output_dir weights/finetuning_uai/vec_final/mask/75/block/8 --in_chan 16 --enc_layers 4 --dec_layers 2 --device cuda:0 --frame_skip 1 --stack_avg 64 --nheads 4 --slow_time 1 --model_scale 't' --kernel 8 --hidden_dim 128 --stride 4 --bbox_loss_coef 0 --lr 4e-4 --lr_min 4e-5 --downstream weights/pretrain_uai/vec_ver3/mask/75/block/8/checkpoint0089.pth --posemodel --unfrozen_layer 4 --weight_decay 0.05 #--model_debug #--alpha 0.$var #--model_debug #--attn_mean #--model_debug --attn_mean #--model_debug #--posemodel  #--model_debug #--resume weights/scratch_convnext16token_downsamplex_vitx/checkpoint.pth #--model_debug
    python main.py --batch_size 64 --epochs 25 --output_dir weights/finetuning_review/channel_wise/mask_75/block_8 --in_chan 16 --enc_layers 4 --dec_layers 2 --device cuda:0 --frame_skip 1 --stack_avg 64 --nheads 4 --slow_time 1 --model_scale 't' --kernel 8 --hidden_dim 128 --stride 4 --bbox_loss_coef 0 --lr 4e-4 --lr_min 4e-5 --downstream weights/pretrain_uai/vec_review/channelwise/mask75/block/8/checkpoint0089.pth --posemodel --unfrozen_layer 4 --weight_decay 0.05 --embtype 'channel_wise' #--model_debug #--alpha 0.$var #--model_debug #--attn_mean #--model_debug --attn_mean #--model_debug #--posemodel  #--model_debug #--resume weights/scratch_convnext16token_downsamplex_vitx/checkpoint.pth #--model_debug
done
