for D in 0-0 1-1 2-2-v2 3-4 5-6; do # loop datasets
    DD="obj-${D}"
    python3 presto/scripts/eval_rrt.py \
        --label=${DD} \
        --data_path=data/presto_cabinet_eval_rrt
done