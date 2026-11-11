# CANDID guidance inspection

`scripts/inspect_candid_guidance.py` compares one Vanilla DP action chunk with the consequence-guided chunk from the same observation and the same diffusion noise.

The denoiser and DDPM scheduler stay in `model/vanilla_dp`. Guidance only adds a predictor gradient after `scheduler.step`. Existing evaluation scripts are unchanged.

Without a simulator, the command reads the recorded observation for the requested scene and writes JSON, NPZ, and plots. It does not invent a robot image.
