from .base import AbstractDataset
from .utils import *

from datetime import date
from pathlib import Path
import pickle
import shutil
import tempfile
import os

import re
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class ML100KDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-100k'

    @classmethod
    def url(cls):  # as of Sep 2023
        return 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README',
                'movies.csv',
                'ratings.csv',
                'users.csv']

    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return

        print("Raw file doesn't exist. Downloading...")
        tmproot = Path(tempfile.mkdtemp())
        tmpzip = tmproot.joinpath('file.zip')
        tmpfolder = tmproot.joinpath('folder')
        download(self.url(), tmpzip)
        unzip(tmpzip, tmpfolder)
        if self.zip_file_content_is_folder():
            tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
        shutil.move(tmpfolder, folder_path)
        shutil.rmtree(tmproot)
        print()

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        meta_raw = self.load_meta_dict()
        
        print(f'Total items in metadata: {len(meta_raw)}')
        
        # BƯỚC 1: Lọc text trước (nhanh)
        if self.args.use_text:
            valid_text_items = set()
            for item_id, meta_info in meta_raw.items():
                if meta_info['text'] is not None and len(meta_info['text']) > 0:
                    valid_text_items.add(item_id)
            print(f'Items with valid text: {len(valid_text_items)}')
            df = df[df['sid'].isin(valid_text_items)]
        else:
            df = df[df['sid'].isin(meta_raw)]  # filter items without meta info
        
        # BƯỚC 2: Lọc triplets (min_uc, min_sc) - giảm số lượng items đáng kể
        print(f'Ratings before filter_triplets: {len(df)}')
        df = self.filter_triplets(df)
        print(f'Ratings after filter_triplets: {len(df)}')
        
        # BƯỚC 3: Densify index - tạo mapping mới
        df, umap, smap = self.densify_index(df)
        
        # BƯỚC 4: Xác định items còn lại sau khi lọc triplets
        remaining_items = set(smap.keys())  # Original item IDs còn lại
        print(f'Items remaining after triplet filtering: {len(remaining_items)}')
        
        # BƯỚC 5: MovieLens không có images, chỉ cảnh báo nếu use_image=True
        if self.args.use_image:
            print('WARNING: MovieLens dataset does not have images!')
            print('All items will be removed if --use_image is enabled.')
            # Không có image nào, loại bỏ tất cả
            df = df[df['sid'].isin([])]
            df, umap, smap = self.densify_index(df)
            print(f'Final items after image filtering: 0')
        train, val, test = self.split_df(df, len(umap))
        meta = {smap[k]: v for k, v in meta_raw.items() if k in smap}
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'meta': meta,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.csv')
        df = pd.read_csv(file_path)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df

    def load_meta_dict(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('movies.csv')
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
        meta_dict = {}
        for row in df.itertuples():
            title = row[2][:-7]  # remove year (optional)
            year = row[2][-7:]

            title = re.sub('\(.*?\)', '', title).strip()
            # the rest articles and parentheses are not considered here
            if any(', '+x in title.lower()[-5:] for x in ['a', 'an', 'the']):
                title_pre = title.split(', ')[:-1]
                title_post = title.split(', ')[-1]
                title_pre = ', '.join(title_pre)
                title = title_post + ' ' + title_pre

            full_title = title + year
            # Store metadata with text (MovieLens doesn't have images)
            meta_dict[row[1]] = {
                'text': full_title.strip() if full_title.strip() else None,
                'image': None,  # MovieLens doesn't have image data
                'title': full_title.strip() if full_title.strip() else None
            }
        return meta_dict
