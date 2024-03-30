import os
from utils import download_from_drive, unzip
import pandas as pd
import numpy as np
from typing import Union
from tqdm import tqdm
import ast


class GetDataset():
    already_unzipped = False

    def __init__(self,
                 id: str = "1Tb18Pz3Z1E5dGurvnE3Zv8z0516WHRGn",
                 seed: int = 42,
                 CreateDataFrame:bool = False,
                 Visualize_DataFrame:bool = False,
                 **args):

        self.seed = seed
        self.__id = id
        self.__folder = os.path.join(os.path.dirname(__file__),
                                     'Dataset')
        self.visualize_dataframe = Visualize_DataFrame
        self.createDataframe = CreateDataFrame
        self.args = args

        if not GetDataset.already_unzipped:
            self.__set_env()
            GetDataset.already_unzipped = True

    def Outputdata(self) -> tuple:
        dff = self.ConcatDataset(self.createDataframe)

        if self.visualize_dataframe:
            mode = self.args.get("mode","head")
            self.Visualizate(
                df=dff, mode=mode, cant=10)

        if isinstance(dff, list):
            datas_arrays = []
            targets_arrays = []

            for dfs in dff:
                N = self.__len__(dfs)
                Predictors = 2
                targets = 1
                dfs['x1'] = dfs['x1'].apply(ast.literal_eval)
                dfs['x2'] = dfs['x2'].apply(ast.literal_eval)
                G_x1 = len(dfs['x1'][0])
                G_x2 = len(dfs['x2'][0])
                G = max(G_x1, G_x2)

                data_array = np.zeros((N, Predictors, G))
                target_array = np.zeros((N, targets))

                for i in range(N):
                    data_array[i, 0, :G] = dfs['x1'][i]
                    data_array[i, 1, :G] = dfs['x2'][i]
                    target_array[i, 0] = float(dfs['band_gap_mean'][i])
                target_array = target_array.reshape(target_array.shape[0])
                datas_arrays.append(data_array)
                targets_arrays.append(target_array)
            return datas_arrays, targets_arrays

        elif isinstance(dff, pd.DataFrame):
            N = self.__len__(dff)
            Predictors = 2
            targets = 1
            dff['x1'] = dff['x1'].apply(ast.literal_eval)
            dff['x2'] = dff['x2'].apply(ast.literal_eval)
            G_x1 = len(dff['x1'][0])
            G_x2 = len(dff['x2'][0])
            G = max(G_x1, G_x2)

            data_array = np.zeros((N, Predictors, G))
            target_array = np.zeros((N, targets))

            for i in range(N):
                data_array[i, 0, :G] = dff['x1'][i]
                data_array[i, 1, :G] = dff['x2'][i]
                target_array[i, 0] = float(dff['band_gap_mean'][i])
            target_array = target_array.reshape(target_array.shape[0])
            return data_array, target_array

    def __set_env(self):
        destination_path_zip = os.path.join(self.__folder,
                                            'Bandgap.zip')
        os.makedirs(self.__folder, exist_ok=True)
        download_from_drive(self.__id, destination_path_zip)
        unzip(destination_path_zip, self.__folder)

    def ConcatDataset(self, createDataframe: bool = False) -> Union[list, pd.DataFrame]:
        dfs = []
        for df_name in os.listdir(os.path.join(self.__folder, 'Bandgap')):
            if df_name == 'dataset_Paper1_Fluorite.csv':
                df = pd.read_csv(os.path.join(
                    self.__folder, 'Bandgap', df_name))
                df = df.drop('band_gap_std', axis=1)
                dfs.append(df)
            elif df_name == 'Dataset.csv':
                pass
            else:
                dfs.append(pd.read_csv(os.path.join(
                    self.__folder, 'Bandgap', df_name)))
        if createDataframe:
            dff = pd.concat(dfs, ignore_index=True)
            dff = dff.drop(dff.columns[0], axis=1)
            dff.to_csv(os.path.join(
                self.__folder, 'Bandgap', 'Dataset.csv'))
            return dff
        else:
            return dfs

    def __len__(self, df):
        return len(df)

    def Visualizate(self,
                    df: Union[list, pd.DataFrame],
                    mode: str = "head",
                    cant: int = 5):
        if isinstance(df, list):
            for dfs in df:
                if mode == "head":
                    print(dfs.head(cant))
                elif mode == "tail":
                    print(dfs.tail(cant))
                else:
                    raise ValueError(
                        "Modo no valido. Seleccione 'head' o 'tail'.")
        elif isinstance(df, pd.DataFrame):
            if mode == "head":
                print(df.head(cant))
            elif mode == "tail":
                print(df.tail(cant))
            else:
                raise ValueError("Modo no valido. Seleccione 'head' o 'tail'.")


if __name__ == '__main__':
    data, target = GetDataset().Outputdata()
    print(data[1].shape)
    print(target[1].shape)
