import xarray as xr

import pooch

bathymeter_mo_ps43 = xr.open_dataset(
    pooch.retrieve(
        url="https://github.com/pyNEMO/sample-datasets/raw/main/bathymeter_mo_ps43.nc",
        known_hash="c8e6b6e185bc15e89e19a860acc79ed9fee335de0c9113af76dac82438297b45",
    )
)
