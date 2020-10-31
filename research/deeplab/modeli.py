export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/deeplab 
# python deeplab/train.py \
#     --logtostderr \
#     --training_number_of_steps=100000 \
#     --train_split="train" \
#     --model_variant="xception_65" \
#     --atrous_rates=6 \
#     --atrous_rates=12 \
#     --atrous_rates=18 \
#     --output_stride=16 \
#     --decoder_output_stride=4 \
#     --train_crop_size=513 \
#     --train_crop_size=513 \
#     --train_batch_size=2 \
#     --dataset="modanet" \
#     --tf_initial_checkpoint='/media/disk_d/WorkingDir/fashion/research/deeplab/traindeeplab/deeplabv3_pascal_train_aug/model.ckpt' \
#     --train_logdir='/media/disk_d/WorkingDir/fashion/research/deeplab/results' \
#     --dataset_dir='/media/disk_d/WorkingDir/fashion/datasets/modanet/tfrecord'
python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size=400 \
    --train_crop_size=600 \
    --train_batch_size=4 \
    --dataset="modanet" \
    --tf_initial_checkpoint='/home/nina//fashion/research/deeplab/traindeeplab/model_50_4/model.ckpt' \
    --train_logdir='/home/nina/fashion/research/deeplab/results/model_100_4/train' \
    --dataset_dir='/home/nina/fashion/datasets/modanet/tfrecord2'

python deeplab/eval.py \
    --logtostderr \
    --eval_split="trainval" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size=600 \
    --eval_crop_size=600 \
    --dataset="modanet" \
    --checkpoint_dir='/home/nina/fashion/research/deeplab/results/model_100_4/train_all' \
    --eval_logdir='/home/nina/fashion/research/deeplab/results/model_100_4' \
    --dataset_dir='/home/nina/fashion/datasets/modanet/tfrecord2'

python deeplab/vis.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size=600 \
    --vis_crop_size=600 \
    --dataset="modanet" \
    --colormap_type="pascal" \
    --checkpoint_dir='/home/nina/fashion/research/deeplab/results/model_100_4/train_all/model.ckpt-484395' \
    --vis_logdir='/home/nina/fashion/research/deeplab/results/model_100_4/vis_model.ckpt-484395' \
    --dataset_dir='/home/nina/fashion/datasets/modanet/tfrecord2'
##############
#vrjetno bi blo treba dat še fine_tune_batch_norm =True pa batch_size>12
python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000000 \
    --train_split="train" \
    --model_variant="mobilenet_v2" \
    --output_stride=16 \
    --train_crop_size=400 \
    --train_crop_size=600 \
    --fine_tune_batch_norm=true \
    --train_batch_size=16 \
    --dataset="modanet" \
    --tf_initial_checkpoint='/home/nina/fashion/research/deeplab/traindeeplab/deeplabv3_mnv2_cityscapes_train/model.ckpt' \
    --train_logdir='/home/nina/fashion/research/deeplab/results/mobilenet_model_100_4/train' \
    --dataset_dir='/home/nina/fashion/datasets/modanet/tfrecord'
python deeplab/eval.py \
    --logtostderr \
    --eval_split="trainval" \
    --model_variant="mobilenet_v2" \
    --output_stride=16 \
    --eval_crop_size=600 \
    --eval_crop_size=600 \
    --dataset="modanet" \
    --checkpoint_dir='/home/nina/fashion/research/deeplab/results/mobilenet_model_100_4/train' \
    --eval_logdir='/home/nina/fashion/research/deeplab/results/mobilenet_model_100_4/eval' \
    --dataset_dir='/home/nina/fashion/datasets/modanet/tfrecord'
python deeplab/vis.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="mobilenet_v2" \
    --output_stride=16 \
    --vis_crop_size=600 \
    --vis_crop_size=600 \
    --dataset="modanet" \
    --colormap_type="pascal" \
    --checkpoint_dir='/home/nina/fashion/research/deeplab/results/mobilenet_model_100_4/train_tmp/model.ckpt-192213' \
    --vis_logdir='/home/nina/fashion/research/deeplab/results/mobilenet_model_100_4/vis_192213' \
    --dataset_dir='/home/nina/fashion/datasets/modanet/tfrecord'
tensorboard --logdir='deeplab/results/mobilenet_model_100_4/train'

#uporabimo anotacije z več kategorijami
python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000000 \
    --train_split="train" \
    --model_variant="mobilenet_v2" \
    --output_stride=16 \
    --train_crop_size=400 \
    --train_crop_size=600 \
    --fine_tune_batch_norm=true \
    --train_batch_size=16 \
    --dataset="modanet" \
    --tf_initial_checkpoint='/home/nina/fashion/research/deeplab/traindeeplab/deeplabv3_mnv2_cityscapes_train/model.ckpt' \
    --train_logdir='/home/nina/fashion/research/deeplab/results/mobilenet_model_100_16_full/train' \
    --dataset_dir='/home/nina/fashion/datasets/modanet/tfrecord3'


python deeplab/train.py --logtostderr --training_number_of_steps=30000000 --train_split="train" --model_variant="mobilenet_v2" --output_stride=16 --train_crop_size=400 --train_crop_size=600 --fine_tune_batch_norm=true --train_batch_size=16 --dataset="modanet" --tf_initial_checkpoint='/home/nina/fashion/research/deeplab/traindeeplab/deeplabv3_mnv2_cityscapes_train/model.ckpt' --train_logdir='/home/nina/fashion/research/deeplab/results/mobilenet_model_100_16_full/train' --dataset_dir='/home/nina/fashion/datasets/modanet/tfrecord3'


python deeplab/eval.py \
    --logtostderr \
    --eval_split="trainval" \
    --model_variant="mobilenet_v2" \
    --output_stride=16 \
    --eval_crop_size=600 \
    --eval_crop_size=600 \
    --dataset="modanet" \
    --checkpoint_dir='/home/nina/fashion/research/deeplab/results/mobilenet_model_100_16_full/train' \
    --eval_logdir='/home/nina/fashion/research/deeplab/results/mobilenet_model_100_16_full/eval' \
    --dataset_dir='/home/nina/fashion/datasets/modanet/tfrecord3'
python deeplab/vis.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="mobilenet_v2" \
    --output_stride=16 \
    --vis_crop_size=600 \
    --vis_crop_size=600 \
    --dataset="modanet" \
    --colormap_type="pascal" \
    --checkpoint_dir='/home/nina/fashion/research/deeplab/results/mobilenet_model_100_16_full/train_tmp/model.ckpt-192213' \
    --vis_logdir='/home/nina/fashion/research/deeplab/results/mobilenet_model_100_16_full/vis_192213' \
    --dataset_dir='/home/nina/fashion/datasets/modanet/tfrecord3'
tensorboard --logdir='deeplab/results/mobilenet_model_100_16_full/train'

