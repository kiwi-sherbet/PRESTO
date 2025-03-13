#!/usr/bin/env bash

set -euxo pipefail

RUN_IDS='run-326' # replace with your own if you'd like
CUDA_DEVICE=0
DIFF_STEP=64
EXPAND=4 # number of seeds to sample for trajopt
SCRIPT_DIR="$( cd "$( dirname $(realpath "${BASH_SOURCE[0]}") )" && pwd )"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

pushd ${REPO_ROOT}
for RUN_ID in ${RUN_IDS}; do
    LOAD_DIR="data/${RUN_ID}"
    for D in 0-0 1-1 2-2-v2 3-4 5-6; do # loop datasets
        DD="obj-${D}"
        for OPT_STEP in 0 8; do # optimizer iterations
            for GUIDE_STEP in 0 1; do # guidance steps
                for DIF in 1; do # enable diffusion
                    if [ ${OPT_STEP} = 0 ]; then # enable optimization
                        OPT=0
                    else
                        OPT=1
                    fi

                    TAG="NEW2-O${OPT}-D${DIF}-P0-N${OPT_STEP}"
                    for d in data/etude_cabinet_eval/${DD}/merge-cloud-v2; do
                        KEY=$(basename $(dirname $d))
                        CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python3 presto/scripts/evaluate.py train.batch_size=1 device=cuda show=none \
                            load_dir="${LOAD_DIR}" \
                            pipeline.expand=${EXPAND} \
                            pipeline.cost.force_reset=0 pipeline.cost.relabel=0 pipeline.cost.sweep=1 pipeline.cost.margin=0.01 \
                            data.shelf.dataset_dir="${d}" \
                            pipeline.use_cached=1 \
                            data.shelf.prim=1 \
                            pipeline.optimize=${OPT} pipeline.opt_margin=0.01 pipeline.n_opt_step=${OPT_STEP} \
                            num_eval=256 shuffle=0 pipeline.apply_constraint=1 \
                            skip_filtered=1 \
                            pipeline.init_type=random pipeline.cond_type=data \
                            pipeline.n_denoise_step=${DIF} \
                            pipeline.n_guide_step=${GUIDE_STEP} pipeline.guide_start=256 \
                            load_normalizer=ckpt diff_step=${DIFF_STEP} \
                            eval_save_path="${RUN_ID}-${TAG}-${KEY}-S${DIFF_STEP}-G${GUIDE_STEP}-C1.pkl" \
                            2>&1 | tee -a "${RUN_ID}-${TAG}-${KEY}-S${DIFF_STEP}-G${GUIDE_STEP}-C1.txt";
                    done
                done
            done
        done
    done
done
popd
