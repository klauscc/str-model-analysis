#================================================================
#   God Bless You. 
#   
#   author: klaus
#   created date: 2020/07/10
#   description: 
#
#================================================================

name=`basename "$0"`
cur_dir=`dirname "$0"`

python -m str.ctc.train \
    --workspace ~/stor6/workspace/intern2020summer/ctc/$name \
    --config $cur_dir/../config.yml \
    --gpu_id 1,2 \
