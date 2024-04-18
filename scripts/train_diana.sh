DA_PAIR="clipart2sketch" # "sketch2painting" "clipart2quickdraw" "real2painting" "real2sketch" "real2clipart" 
python train.py --cfg_file config/domainnet/clipart2sketch.yml -a GMM -d self_ft

python train.py --cfg_file config/domainnet/clipart2sketch.yml -a uniform

python train.py --cfg_file config/domainnet/clipart2sketch.yml -a CLUE -d mme