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

python -m str.ctc.train \
    --workspace ~/stor5/workspace/intern2020summer/ctc/$name \
    --config $cur_dir/../config.yml \
    --resume 0 \
