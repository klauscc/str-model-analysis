#================================================================
#   God Bless You. 
#   
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2020/07/10
#   description: 
#
#================================================================

name=`basename "$0"`
cur_dir=`dirname "$0"`

python -m str.semantic_reasoning.train \
    --workspace ~/stor5/workspace/intern2020summer/semantic_reasoning/$name \
    --config $cur_dir/../config.yml \
    --multi_scales False \
    --gpu_id 3,4,7 \
