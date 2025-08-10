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



# write your code below
python -u main.py --loss_type bce --device cpu
