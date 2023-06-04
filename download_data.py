from pathlib import Path
import io
import requests
import zipfile

DATA_URL = 'https://www.envidat.ch/dataset/e1758773-8065-42ef-bdce-d7bfba6d19da/resource/780cd0c3-1478-443a-ba17-af1daba34f9d/download/swe2hs_calibration_and_validation_data.zip'
DATA_DIR = Path(__file__).parent.resolve() / "data" / ".model_input_data_cache"

def download_data_from_envidat_repository():
    print(f"downloading data from https://doi.org/10.16904/envidat.394 and saving in \n {DATA_DIR}")
    r = requests.get(DATA_URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(DATA_DIR)

def main():
    if DATA_DIR.exists():
        print(f"using cached data from {DATA_DIR}")
        pass
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        download_data_from_envidat_repository()
    return None

