for MAX_RUNTIME in 1 2 4 8 16 32 64 128 500; do # loop max runtime
    for D in 0-0 1-1 2-2 3-4; do # loop datasets
        DD="obj-${D}"
        python3 presto/scripts/eval_rrt.py \
            --label=${DD} \
            --data_path=data/presto_cabinet_eval_rrt \
            --save_path=data/eval/bi-rrt \
            --max_runtime=${MAX_RUNTIME}
    done
done