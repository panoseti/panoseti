## ADC to P.E. Dataflow

### Collect configuration-files
0. Find the panoseti software commit hash `H` from `sw_info.json` in the target observing run directory.
2. Fetch the following calibration files from commit `H` in the panoseti software:
   - Calibration: `control/quabos/*.json`
   - Pulse-height baselines: `/path/to/obs_*/quabo_ph_baselines.json`
   - Quabo uids 

## Converting the `ph256` PFF Data Product from ADC to P.E. 
Apply the following steps to convert pulse-height frame data from raw ADC units into photoelectron units

0. Re-index pixels to convert from the hardware pixel coordinates (bga or qfp) into x-y spatial pixel coordinates ([pixel_indexing docs](https://github.com/panoseti/panoseti/wiki/Pixel-indexing)).
1. Use `pff` to read the 1D array, then use `numpy` to reshape and cast to `shape=(16,16)` and `dtype=np.int16`, respectively.
2. Read baselines from `quabo_baselines.json`. If necessary, apply the re-index operation to these values. Then cast the 1D array into a `shape=(16,16)` `dtype=np.int16` array.
3. Apply the detector-specific linear transform to the ADC-valued pixels to convert them to p.e. units ([quabo calibration docs](https://github.com/panoseti/panoseti/wiki/Configuration-files#quabo-calibration-quabo_calib_uidjson)).

For an observing run, transformations 0-3 are deterministic for every `ph256` PFF frame a given Quabo $q$ produces.

### Formalizing the `ph256` ADC to P.E. Transformation with Vectorizable Operations
#### Background 
Let $\sigma_q^{-1}: I^\prime \rightarrow I$ denote pixel permutation for Quabo $q$ from [hardware-encoded images](https://github.com/panoseti/panoseti/wiki/Pixel-indexing) to "conventional" images,
where the pixel coordinates $(i,j)$ in $I$ are defined by the following image coordinate convention:
> The origin (0,0) is in the top-left corner of the image, the x-coordinate represents the column number, increasing from left to right, and the y-coordinate represents the row number, increasing from top to bottom ([link](https://math.hws.edu/graphicsbook/c2/s1.html)).

Quabo firmware applies the $\sigma_q^{-1}$ mapping and Hashpipe performs image rotations, so the `ph256` PFF frames from a given module are streams of $I_q$ images with the same module-level orientation 
([pixel indexing docs](https://github.com/panoseti/panoseti/wiki/Pixel-indexing#module-coordinates), [Hashpipe docs](https://github.com/panoseti/panoseti/wiki/Data-recorder:-compute-thread#compute-thread-description), [firmware v11.7 docs](https://github.com/panoseti/quabo_firmware/blob/85d1051460be607d8c5c33db4085a42ade44782c/quabo_master/ReadMe.txt#L44)).

#### Pixel-level ADC to P.E. Transformation
Given a pixel $p_{ij}^{(q)}$ at index $(i,j)$ in a `ph256` frame $I_q$ from Quabo $q$ with scalar pulse-height baseline $b_{ij}^{(q)}$, gain $g_{ij}^{(q)}$, and ADC conversion coefficients $n_{ij}^{(q)}$ and $m_{ij}^{(q)}$, 
the [Quabo Calibration](https://github.com/panoseti/panoseti/wiki/Configuration-files#quabo-calibration-quabo_calib_uidjson) documentation states that the relationship between the ADC-unit value $p_{ij}^{(q)} + b_{ij}^{(q)}$ 
and p.e.-unit value $f\left(p_{ij}^{(q)}\right)$ is given by:`n, m: ADC = m * gain * PE_threshold + n`, 
or
$$p_{ij}^{(q)} + b_{ij}^{(q)} = m_{ij}^{(q)} \cdot g_{ij}^{(q)} \cdot f\left(p_{ij}^{(q)}\right) + n_{ij}^{(q)}$$

Hence, the ADC to p.e. transformation $f$ is given by
$$f\left(p_{ij}^{(q)}\right) = \left((p_{ij}^{(q)} + b_{ij}^{(q)}) - n_{ij}^{(q)}\right) / \left(m_{ij}^{(q)} \cdot g_{ij}^{(q)}\right)$$

#### Image-level ADC to P.E. Transformation
Define the following symbols for a Quabo $q$, where each matrix has dimension $16\times16$ with calibration values organized according to "conventional" image coordinates.
- $B_q$ = pixel baseline matrix, after any $\sigma_q^{-1}$ transformations and rotations applied to values from `quabo_ph_baselines.json` (docs: N/A). 
- $G_q$ = `pixel_gain_delta` matrix, from `quabo_calib_UID.json` ([docs](https://github.com/panoseti/panoseti/wiki/Configuration-files#quabo-calibration-quabo_calib_uidjson)).
- $N_q$ = `n` coefficient block matrix for each detector region from `quabo_calib_UID.json` ([docs](https://github.com/panoseti/panoseti/wiki/Configuration-files#quabo-calibration-quabo_calib_uidjson)).
- $M_q$ = `m` coefficient block matrix for each detector region from `quabo_calib_UID.json`([docs](https://github.com/panoseti/panoseti/wiki/Configuration-files#quabo-calibration-quabo_calib_uidjson)).

Then, the ADC to p.e. transformation for each `ph256` PFF image $I_q$ from Quabo $q$ is given by:

$$
\begin{align} 
f(I_q) &= \left(I_q + B_q - N_q \right) \oslash (M_q \odot G_q) \\
\end{align}
$$

where $\oslash$ denotes element-wise division and $\odot$ denotes element-wise multiplication.

#### Development notes
Some challenges in implementing this transformation at scale include the following:
1. Automatically and reliably gathering all configuration files used during an observing run, including locally from the observing run directory and from specific version-controlled files across (potentially) multiple GitHub repositories. (Use [Bazel](https://bazel.build/) for this?) 
2. Constructing $\sigma_q^{-1}$, $B_q$, $G_q$, $N_q$, and $M_q$ from the various configuration files.
3. Writing code to harness the pixel-level parallelism (write a cuda kernel?).


## Dataframe Schemas (WIP)

The following documentation describes the `pandas` dataframe schemas that are used to organize the disparate configuration file information 
and construct the $\sigma_q^{-1}$, $B_q$, $G_q$, $N_q$, and $M_q$ targets. 

### `quabo_install_df` schema
Each record represents a Quabo used in a specific observing run.

Columns:
- `dome` = dome name from `obs_config.json` file.
- `module_ip_addr` = `ip` address of the module, unique for a given observing run.
- `mobo_serial_no` = module board serial number on which this Quabo was installed.
- `quabo_uid` = UID of the Quabo hardware.
- `quabo_num` = spatial position of the Quabo in its module.
- `detector_overvoltage` = overvoltage setting used in observing run.

Primary keys: (`quabo_uid`)

Dependencies:
- `obs_config.json`
- `quabo_uids.json`


### `quabo_info_df` schema
Each record represents a unique Quabo board known to the panoseti software.

Columns:
- `quabo_uid`= UID of the Quabo hardware.
- `board_version` = board version, one of two values `{"qfp", "bga"}`.
- `serialno_str` = string used in `quabo_info.json` file. e.g. "SN019".
- `serialno` = parsed serial number from `serialno_str`.
- `detector_serialno_i` = serial number of the ith detector array, for `i = 0, 1, 2, 3`.

Primary keys: (`quabo_uid`)

Dependencies:
- `control/quabos/quabo_info.json`

### detector_calibration_df schema
Each record represents a detector-level calibration 

(TODO)

Dependencies:
- `quabo_ph_baseline.json`