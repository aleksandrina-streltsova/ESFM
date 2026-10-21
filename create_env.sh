conda install -c conda-forge numcodecs
conda install conda-forge::asciimatics
pip install zarr==3.1.5
pip install xarray==2025.12.0
pip install pandas jupyter matplotlib scipy h5py netCDF4 tqdm geopandas
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install cmocean cartopy scores esgf-pyclient huggingface-hub einops
pip install --quiet "lightning==2.5.1"
conda install conda-forge::tmux
pip install timm  # model code (esfm.model.util); on a CPU-only torch use the matching CPU torchvision wheel:
# pip install --force-reinstall --no-deps torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
