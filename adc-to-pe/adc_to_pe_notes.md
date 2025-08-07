## ADC to P.E. Dataflow

### Collect configuration-files
0. Find the panoseti software commit hash `H` from `sw_info.json` in the target observing run directory.
2. Fetch the following calibration files from commit `H` in the panoseti software:
   - Calibration: `control/quabos/*.json`
   - Pulse-height baselines: `/path/to/obs_*/quabo_ph_baselines.json`
   - Quabo uids 

## Converting the `ph256` PFF Data Product from ADC to P.E. 
Apply the following steps to convert pulse-height frame data from raw ADC units into photoelectron units

0. Re-index pixels to convert from the hardware pixel coordinates (bga or qfp) into x-y spatial pixel coordinates. (These transformations are likely handled by the firmware + production software.)
1. Use `pff` to read the 1D array, then use `numpy` to reshape and cast to `shape=(16,16)` and `dtype=np.int16`, respectively.
2. Read baselines from `quabo_baselines.json`. If necessary, apply the re-index operation to these values. Then cast the 1D array into a `shape=(16,16)` `dtype=np.int16` array.
3. Apply the detector-specific linear transform to the ADC-valued pixels to convert them to p.e. units.

For an observing run, transformations 0-3 are deterministic for every `ph256` PFF frame a given quabo `q` produces.

### Formalizing the ADC to P.E. Transformation with Vectorizable Operations
For an ADC-valued pixel $p$ with baseline $b$, and adc conversion coefficients $n$ and $m$ from `quabo_calib_X.json`, the ADC to p.e. transformation $f$ is given by
$$f(p) = ((p + b) - n) / m.$$

Define the following symbols for a given quabo `q`, where the matrices and images are $16\times16$ 
- $\sigma_q: I \rightarrow I^\prime$ is the pixel permutation mapping hardware images to "spatial images" where (0, 0) is the pixel in the top left corner and (15, 15) is the pixel in the bottom right corner.
- $B$ = pixel baseline matrix, after any $\sigma_q$ transformations and rotations. 
- $N$ = `n` coefficient block matrix for each detector region from `quabo_calib_X.json`.
- $M$ = `m` coefficient block matrix for each detector region from `quabo_calib_X.json`.

Then the ADC to p.e. transformation for a `ph256` PFF image $I$ is given by:

$$
\begin{align} 
f(I) &= \left((\sigma_q(I) + B) - N \right) \oslash M \\
     &= (\sigma_q(I)\oslash M) + \left(B - N \right) \oslash M \\
     &= (\sigma_q(I)\oslash M) + C
\end{align}
$$

where $\oslash$ denotes element-wise division and $C = \left(B - N \right) \oslash M$.

The main challenges in performing transformation are:
1. Constructing $\sigma_q$, $B$, $N$, and $M$ from the various configuration files, both local to the run and externally version-controlled in the panoseti software repo.
2. Writing code to harness the pixel-level parallelism (write a cuda kernel?).


## Dataframe Schemas

### quabo_install_df schema
Each record represents a quabo used in a specific observing run.

Columns:
- `dome` = dome name from `obs_config.json` file.
- `module_ip_addr` = `ip` address of the module, unique for a given observing run.
- `mobo_serial_no` = module board serial number on which this quabo was installed.
- *`quabo_uid`* = UID of the quabo hardware.
- `quabo_num` = spatial position of the quabo in its module.
- `detector_overvoltage` = overvoltage setting used in observing run.

Primary keys: (`quabo_uid`)

Dependencies:
- `obs_config.json`
- `quabo_uids.json`


### `quabo_info_df` schema
Each record represents a unique quabo board known to the panoseti software.

Columns:
- *`quabo_uid`* = UID of the quabo hardware.
- `board_version` = board version, one of two values `{"qfp", "bga"}`.
- `serialno_str` = string used in `quabo_info.json` file. e.g. "SN019".
- `serialno` = parsed serial number from `serialno_str`.
- `detector_serialno_i` = serial number of the ith detector array, for `i = 0, 1, 2, 3`.

Primary keys: (`quabo_uid`)

Dependencies:
- `control/quabos/quabo_info.json`

### detector_calibration_df schema
Each record represents a detector-level calibration 


Dependencies:
- `quabo_ph_baseline.json`