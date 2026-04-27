import os
import h5py
import torch
from torch.utils.data import Dataset

from utils.utils import read_json_file


class EmbeddingDataset(Dataset):

    def __init__(self, args, tokenizer, dataset_type):

        self.__slides = None
        self.__reports = None
        self.__tokenizer = tokenizer
        self.__max_seq_length = args.max_seq_length
        self.__data_path_patch = args.data_path_patch
        self.get_slides_brca(dataset_type, args.reports_json_path)
        self.max_fea_length = args.max_fea_length

    def get_slides(self, dataset_type, reports_json_path):
        reports = read_json_file(reports_json_path)[dataset_type]
        self.__reports = {report['id'].split('.')[0]: report['report'] for report in reports}

        files = os.listdir(self.__data_path_patch)
        self.__slides = list(self.__reports.keys())

        print(f'dataset_type: {dataset_type} self.__slides: {len(self.__slides)}')
        self.__slides = [file.split('.')[0] for file in files if file.split('.')[0] in self.__slides]
        print(f'dataset_type: {dataset_type}, slides: {len(self.__slides)}')

        # print(
        #     f'dataset_type: {dataset_type}, files: {files[:4]}, files_1: {files_1[:4]}, files_2: {files_2[:4]} slides: {self.__slides[:4]}')


    def get_slides_brca(self, dataset_type, reports_json_path):
        reports = read_json_file(reports_json_path)[dataset_type]
        self.__reports = {report['id']: report['report'] for report in reports}

        files = os.listdir(self.__data_path_patch)
        # files_1 = os.listdir(self.__data_path_patch)
        # files_2 = os.listdir(self.__data_path_concept)

        slides = [slide for slide in self.__reports.keys() if
                  slide != 'TCGA-A2-A1G0-01Z-00-DX1.9ECB0B8A-EF4E-45A9-82AC-EF36375DEF65']

        print(f'dataset_type: {dataset_type} self.__slides: {len(slides)}')

        self.__slides = ['.'.join(file.split('.')[:-1]) for file in files if
                          '.'.join(file.split('.')[:-1]) in slides]
        print(
            f'dataset_type: {dataset_type}, files: {len(files)}, slides: {len(self.__slides)}')

        # print(
        #     f'dataset_type: {dataset_type}, files: {files[:4]}, files_1: {files_1[:4]}, files_2: {files_2[:4]} slides: {self.__slides[:4]}')

    def __len__(self):
        return len(self.__slides)

    def __getitem__(self, idx):
        slide_id = self.__slides[idx]

        report_text = self.__reports[slide_id]
        report_ids = self.__tokenizer(report_text)
        seq_length = min(len(report_ids), self.__max_seq_length)

        if len(report_ids) > self.__max_seq_length:
            report_ids = report_ids[:self.__max_seq_length]
        elif len(report_ids) < self.__max_seq_length:
            padding = [0] * (self.__max_seq_length-len(report_ids))
            report_ids.extend(padding)

        report_masks = [1] * seq_length + [0] * (self.__max_seq_length - seq_length)

        with h5py.File(f'{self.__data_path_patch}/{slide_id}.h5', "r") as h5_file:
            embeddings_np = h5_file["features"][:]
            patch_embedding = torch.tensor(embeddings_np)
            patch_embedding = patch_embedding[:self.max_fea_length]

        return slide_id, patch_embedding, report_ids, report_masks
