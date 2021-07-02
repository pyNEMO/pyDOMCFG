import pooch
import xarray as xr

BASE_URL = "https://github.com/pyNEMO/sample-datasets/raw/main"

ds_amm7 = xr.open_dataset(
    pooch.retrieve(
        url="/".join([BASE_URL, "bathymeter_amm7_mo_ps43.nc"]),
        known_hash="c8e6b6e185bc15e89e19a860acc79ed9fee335de0c9113af76dac82438297b45",
    )
)
