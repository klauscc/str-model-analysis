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
    --workspace ~/stor6/workspace/intern2020summer/semantic_reasoning/$name \
    --config $cur_dir/../config.yml \
    --multi_scales False \
    --with_semantic_reasoning True \
    --gpu_id 0,1,2,3,4,5,6,7 \
