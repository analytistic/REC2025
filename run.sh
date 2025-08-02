#!/bin/bash

# 加载环境变量
source base.env

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}

# show ${TRAIN_LOG_PATH}
echo ${TRAIN_LOG_PATH}

# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# write your code below
python -u main.py --device cpu --mm_emb_id 81 82
