# #!/bin/bash
# pip install toml

# # show ${RUNTIME_SCRIPT_DIR}
# echo ${RUNTIME_SCRIPT_DIR}
# # enter train workspace
# cd ${RUNTIME_SCRIPT_DIR}



# # 检查并解压 all.zip
# if [ -f "all.zip" ]; then
#     echo "Found all.zip, extracting to current directory..."
#     unzip -o all.zip
#     if [ $? -eq 0 ]; then
#         echo "✓ Successfully extracted all.zip"
#         rm all.zip  # 删除压缩包节省空间
#     else
#         echo "✗ Failed to extract all.zip"
#         exit 1
#     fi
# else
#     echo "No all.zip found, assuming files are already in place"
# fi

# python -u main.py --loss_type bce --device cuda --num_blocks 1 --num_heads 1 --hidden_units 32

# # write your code below
# python -u main.py --loss_type bce --device cpu

# python -u main.py --loss_type bce --device cpu --num_blocks 4 --num_heads 4 --hidden_units 64
# python -u main.py --loss_type inbatch_infonce --device cuda --num_blocks 4 --num_heads 4 --hidden_units 64
# python -u main.py --loss_type infonce --device cpu
# python -u main.py --loss_type infonce --device cpu
# python -u main.py --loss_type cosine_triplet --device cpu --num_blocks 1 --num_heads 1 --hidden_units 32
python -u main.py --loss_type ado_infonce --device cpu --num_blocks 1 --num_heads 1 --hidden_units 32 --num_epochs 60