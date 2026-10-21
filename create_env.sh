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
# PhysMetrics.Weather (physical-consistency metrics), cloned next to this repo; its branch land-water-balance has the land checks.
# The pins keep the requirements of lightning 2.5.1 (packaging<25, fsspec<2026), which PhysMetrics' dependencies would otherwise upgrade.
git clone https://github.com/Emmakast/PhysMetrics.Weather.git /users/$USER/PhysMetrics.Weather
pip install -e /users/$USER/PhysMetrics.Weather "packaging<25" "fsspec[http]<2026" "gcsfs<2026"
