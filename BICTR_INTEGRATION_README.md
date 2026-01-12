# BICTR Channel Model Integration

This document describes the BICTR (Lunar Wireless Channel Model) integration into OpenAirInterface5G.

## Overview

BICTR has been integrated into the OpenAirInterface channel model system with full configuration support through the existing channelmod configuration framework.

## What Was Integrated

1. **BICTR Configuration Structure** (`bictr_config_t`) - Added to `openair1/SIMULATION/TOOLS/sim.h`
2. **BICTR Parameter Definitions** - Added 13 BICTR-specific parameters to the channelmod configuration system
3. **Configuration Storage and Retrieval** - BICTR configs are stored and retrieved by model name
4. **Configuration Files** - Created example configuration files for BICTR usage

## Configuration Parameters

All BICTR parameters can be configured in the channelmod section:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bictr_refCount` | 5 | Maximum number of reflectors to generate |
| `bictr_refAttemptPerRing` | 3 | Number of reflector attempts per ring |
| `bictr_ringRadiusMin` | 5.0 | Minimum ring radius in meters |
| `bictr_ringRadiusMax` | 300.0 | Maximum ring radius in meters |
| `bictr_ringRadiusUncertainty` | 15.0 | Random offset from ring radius (meters) |
| `bictr_ringCount` | 10 | Number of rings between min and max |
| `bictr_complexRelPermittivityReal` | 3.0 | Real part of complex permittivity |
| `bictr_complexRelPermittivityRealStd` | 0.5 | Standard deviation of real part |
| `bictr_complexRelPermittivityImag` | 0.1 | Imaginary part of permittivity |
| `bictr_complexRelPermittivityImagStd` | 0.05 | Standard deviation of imaginary part |
| `bictr_horizontalPolarization` | 1 | 1 for horizontal, 0 for vertical |
| `bictr_fadingPaths` | 20 | Number of paths for Rayleigh fading |
| `bictr_fadingDopplerSpread` | 0.0 | Doppler spread in m/s |

## Configuration Files

### 1. `channelmod_bictr.conf`
Base BICTR channel model configuration with two model lists:
- `modellist_bictr` - Standard configuration (fast computation mode)
- `modellist_bictr_aggressive` - Aggressive search mode for highly chaotic terrain

### 2. `gnb.sa.band78.106prb.rfsim.bictr.conf`
Complete gNB configuration file with BICTR channel model enabled.

### 3. `bictr.conf`
Updated to include channelmod section for BICTR.

## Build Instructions

1. **Navigate to build directory:**
   ```bash
   cd /home/chris/Desktop/openairinterface5g-sl-release-1.0/cmake_targets
   ```

2. **Clean previous build (optional):**
   ```bash
   rm -rf ran_build/build/*
   ```

3. **Build the project:**
   ```bash
   ./build_oai -C -w USRP --gNB --nrUE -x
   ```
   
   Or if using a different build system:
   ```bash
   cd ran_build/build
   make -j$(nproc)
   ```

## Run Instructions

### Method 1: Using BICTR Configuration File

```bash
cd /home/chris/Desktop/openairinterface5g-sl-release-1.0
sudo ./cmake_targets/ran_build/build/nr-softmodem \
  -O ci-scripts/conf_files/gnb.sa.band78.106prb.rfsim.bictr.conf \
  --rfsim \
  --rfsimulator.options chanmod \
  --rfsimulator.serveraddr server
```

### Method 2: Using Command-Line Parameter Overrides

```bash
sudo ./cmake_targets/ran_build/build/nr-softmodem \
  -O ci-scripts/conf_files/gnb.sa.band78.106prb.rfsim.conf \
  --rfsim \
  --rfsimulator.options chanmod \
  --channelmod.modellist modellist_bictr \
  --channelmod.modellist_bictr.[0].type BICTR \
  --channelmod.modellist_bictr.[0].bictr_refCount 5 \
  --channelmod.modellist_bictr.[0].bictr_ringRadiusMax 500.0
```

### Method 3: Using Aggressive Search Mode

```bash
sudo ./cmake_targets/ran_build/build/nr-softmodem \
  -O ci-scripts/conf_files/gnb.sa.band78.106prb.rfsim.bictr.conf \
  --rfsim \
  --rfsimulator.options chanmod \
  --channelmod.modellist modellist_bictr_aggressive
```

## Verification

1. **Check logs for BICTR initialization:**
   Look for messages like:
   ```
   BICTR config stored for model rfsimu_channel_enB0
   Model rfsimu_channel_enB0 type BICTR allocated from config file
   ```

2. **Verify channel model is active:**
   Use telnet to check (if telnet server enabled):
   ```bash
   telnet 127.0.0.1 9090
   > channelmod show current
   ```

3. **Monitor channel statistics:**
   Check log files for channel model statistics and path information.

## Troubleshooting

### If BICTR parameters not found:
- Ensure `channelmod_bictr.conf` is included in your main config
- Check that `modellist` points to a list containing BICTR models
- Verify parameter names match exactly (case-sensitive)

### If build fails:
- Ensure `bictr_config_t` structure is defined in `sim.h`
- Check that all BICTR functions are implemented in `random_channel.c`
- Verify includes are correct

### If channel model doesn't apply:
- Ensure `--rfsimulator.options chanmod` is set
- Verify model name matches: `rfsimu_channel_enB0` or `rfsimu_channel_ue0`
- Check that `type = "BICTR"` in config file

## Configuration Modes

### Fast Computation Mode (Default)
- `bictr_refAttemptPerRing = 3`
- `bictr_ringRadiusMax = 300.0`
- `bictr_ringRadiusUncertainty = 15.0`

### Aggressive Search Mode
- `bictr_refAttemptPerRing = 15`
- `bictr_ringRadiusMax = 500.0`
- `bictr_ringRadiusUncertainty = 25.0`

Use aggressive mode for highly chaotic terrain with reduced LOS visibility.

## Files Modified

1. `openair1/SIMULATION/TOOLS/sim.h`
   - Added `bictr_config_t` structure
   - Added BICTR parameter name definitions
   - Extended `CHANNELMOD_MODEL_PARAMS_DESC` with BICTR parameters

2. `openair1/SIMULATION/TOOLS/random_channel.c`
   - Added BICTR config storage and retrieval functions
   - Updated `load_channellist` to read and store BICTR parameters
   - Updated BICTR case in `new_channel_desc_scm`

3. Configuration files:
   - `ci-scripts/conf_files/channelmod_bictr.conf` (new)
   - `ci-scripts/conf_files/gnb.sa.band78.106prb.rfsim.bictr.conf` (new)
   - `ci-scripts/conf_files/bictr.conf` (updated)

## Next Steps

The current implementation provides the configuration framework. The full BICTR algorithm implementation (reflector generation, Rayleigh fading, etc.) from the Python version can be integrated into the `init_bictr_channel_desc` function and the BICTR case in `new_channel_desc_scm` to use the stored configuration parameters.
