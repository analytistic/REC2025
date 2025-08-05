from dotenv import load_dotenv
import os
import json
import pandas as pd
from tqdm import tqdm
load_dotenv(dotenv_path="/Users/alex/project/Rec/rec_2025/base.env")
# import cudf

class model():
    def __init__(self, cfg):
        self.cfg = cfg
        self.CHUNK = cfg.CHUNK
        self.DISK_PIECES = cfg.DISK_PIECES
        self.READ_CT = cfg.READ_CT
        self.PARQUET_SIZE = cfg.PARQUET_SIZE
  


        self.df = self.read_file_to_cache(os.getenv("TRAIN_DATA_PATH"))


    def read_file(self, index):
        # return cudf.DataFrame(self.df[index])
        return pd.DataFrame(self.df[index])

    def read_file_to_cache(self, file_path):
        """
        读取文件并将内容存储到缓存中
        Args:
            file_path: 文件路径
        Returns:
            dict: 文件内容的字典形式
        """
        data = []
        sub_data = []
        parquet_num = 0
        with open(file_path+"/seq.jsonl", 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()
                if line:
                    user_seq = json.loads(line)
                    for item in user_seq:
                        sub_data.append([item[i] for i in (0, 1, 4, 5)])
                    parquet_num += 1
                    if parquet_num == self.PARQUET_SIZE:
                        df = pd.DataFrame(sub_data, columns=['user_id', 'item_id', 'action', 'ts'])
                        df.ts = (df.ts/1000).astype('int32')
                        data.append(df)
                        parquet_num = 0
                        sub_data = []
                
        print(f"读取文件 {file_path} 成功")
        return data
    
    def fit(self):
        for PART in range(self.DISK_PIECES):
            print(f'### PART {PART+1}/{self.DISK_PIECES}\n')

            # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
            # => OUTER CHUNKS
            for j in range((len(self.df)//self.CHUNK)+1):
                a = j*self.CHUNK
                b = min((j+1)*self.CHUNK, len(self.df))
                print(f'Processing files {a} thru {b-1} in groups of {self.READ_CT}...')

                # => INNER CHUNKS
                for k in range(a, b, self.READ_CT):
                    # READ FILE
                    df = [self.read_file(self.df[k])]
                    for i in range(1, self.READ_CT):
                        if k+i < b: df.append(self.read_file(self.df[k+i]))
                    df = pd.concat(df, ignore_index=True, axis=0)
                    df = dd.from_pandas(df, npartitions=DASK_MULTIPROCESS)
                    # df = cudf.concat(df, ignore_index=True, axis=0)
                    df = df.sort_values(['session', 'ts'], ascending=[True, False])
                    # df = df.reset_index(drop=True) 我也不知到，反转dasksort后面不能reset_index
                    df['n'] = df.groupby('session').cumcount()
                    df = df.loc[df.n<30].drop('n', axis=1) # 找出最近30次交互aid
                    # CREATE PAIRS
                    df = df.merge(df, on='session')
                    df = df.loc[((df.ts_x - df.ts_y).abs() < 24 * 60 * 60) &(df.aid_x != df.aid_y) ] # 只保留最近24小时内的交互
                    # MEMORY MANAGEMENT COMPUTE IN PARTS
                    df = df.loc[
                        (df.aid_x >= PART*SIZE) & (df.aid_x < (PART+1)*SIZE)
                    ]
                    # ASSIGN WEIGHTS
                    df = df[['session', 'aid_x', 'aid_y', 'type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
                    df['wgt'] = df.type_y.map(type_weight)
                    df = df[['aid_x', 'aid_y', 'wgt']]
                    df.wgt = df.wgt.astype('float32')
                    df = df.groupby(['aid_x', 'aid_y']).wgt.sum()
                    # COMBINE INNER CHUNKS
                    if k == a: tmp2 = df
                    else: tmp2 = tmp2.add(df, fill_value=0)
                    print(k, ', ', end='')
                print()
                # COMBINE OUTER CHUNKS
                if a==0: tmp = tmp2
                else: tmp = tmp.add(tmp2, fill_value=0)
                del tmp2, df
                gc.collect()
            # CONVERT MATRIX TO DICTIONARY
            tmp = tmp.reset_index()
            tmp = tmp.sort_values(['aid_x', 'aid_y'], ascending=[True, False])
            # SAVE TOP_K
            tmp['n'] = tmp.groupby('aid_x').cumcount()
            tmp = tmp.loc[tmp.n<I2I_TOPK].drop('n', axis=1)
            tmp = tmp.compute()
            tmp.to_parquet(f'top_K_carts_orders_v{VER}_{PART}.parquet', index=False)
            









if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.BaseConfig import LazyConfig
    cfg = LazyConfig("/Users/alex/project/Rec/rec_2025/module/config/itemcf.toml")
    model_cf = model(cfg)
