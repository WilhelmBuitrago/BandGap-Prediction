import os
from utils import download_from_drive, unzip


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

        if not GetDataset.already_unzipped:
            self.__set_env()
            GetDataset.already_unzipped = True

    def __set_env(self):
        destination_path_zip = os.path.join(self.__folder,
                                            'Bandgap.zip')
        os.makedirs(self.__folder, exist_ok=True)
        download_from_drive(self.__id, destination_path_zip)
        unzip(destination_path_zip, self.__folder)


if __name__ == '__main__':
    GetDataset()
