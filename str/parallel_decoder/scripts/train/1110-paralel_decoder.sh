#================================================================
#   Don't go gently into that good night. 
#   
#   author: klaus
#   created date: 2020/11/10
#   description: 
#
#================================================================


name=`basename "$0"`
cur_dir=`dirname "$0"`

workspace=$HOME/stor6/workspace/intern2020summer/parallel_decoder/$name
mkdir -p $workspace
python -m str.parallel_decoder.train \
    --workspace  $workspace \
    --config $cur_dir/../../config_large.yml \
    --multi_scales False \
    --keep_aspect_ratio False \
    --voc_type LOWERCASE \
    --gpu_id 1,2,3,4 \
    --encode_training_mode encoder_only  \
    --label_smooth 0.1 \
