cd $(dirname $0)/..

DATASETS=mnms2
INPUT=./data/
OUTPUT=./results/

batch_size=16

models=msd_umamba
epochs=1
min_epochs=0
start_valid_epoch=1
valid_interval=1

opt=RMSprop
lr=1e-4
momentum=0.9
weight_decay=1e-4

sched=cosine

python -m run.train \
    $DATASETS --input $INPUT --output $OUTPUT \
    --batch_size $batch_size \
    --models $models --epochs $epochs --min_epochs $min_epochs \
    --start_valid_epoch $start_valid_epoch \
    --valid_interval $valid_interval \
    --opt $opt --lr $lr --momentum $momentum --weight_decay $weight_decay \
    --sched $sched
