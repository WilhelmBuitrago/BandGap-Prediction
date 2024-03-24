import os
from utils import download_from_drive, unzip
import pandas as pd


class GetDataset():
    already_unzipped = False

    def __init__(self,
                 id: str = "1Tb18Pz3Z1E5dGurvnE3Zv8z0516WHRGn",
                 seed: int = 42,
                 **args):

        self.seed = seed
        self.__id = id
        self.__folder = os.path.join(os.path.dirname(__file__),
                                     'Dataset')
        visualize_dataframe = args.get("VisualizateDataframe")
        createDataframe = args.get("CreateDataframe", False)
        if not GetDataset.already_unzipped:
            self.__set_env()
            GetDataset.already_unzipped = True
        if createDataframe:
            dff = self.ConcatDataset()
            if visualize_dataframe:
                if len(args["VisualizateDataframe"]) < 2 or len(args["VisualizateDataframe"]) > 2:
                    raise ValueError(
                        "El argumento 'VisualizateDataframe' toma unica y exactamente dos valores [True/False, 'head'/'tail']")
                if args["VisualizateDataframe"][0]:
                    self.Visualizate(
                        df=dff, mode=args["VisualizateDataframe"][1], cant=10)

    def __set_env(self):
        destination_path_zip = os.path.join(self.__folder,
                                            'Bandgap.zip')
        os.makedirs(self.__folder, exist_ok=True)
        download_from_drive(self.__id, destination_path_zip)
        unzip(destination_path_zip, self.__folder)

    def ConcatDataset(self) -> pd.DataFrame:
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
        dff = pd.concat(dfs, ignore_index=True)
        dff = dff.drop(dff.columns[0], axis=1)
        dff.to_csv(os.path.join(
            self.__folder, 'Bandgap', 'Dataset.csv'))

        return dff

    def __len__(self, df):
        return len(df)

    def Visualizate(self,
                    df: pd.DataFrame,
                    mode: str = "head",
                    cant: int = 5):
        if mode == "head":
            print(df.head(cant))
        elif mode == "tail":
            print(df.tail(cant))
        else:
            raise ValueError("Modo no valido. Seleccione 'head' o 'tail'.")


if __name__ == '__main__':
    GetDataset()
