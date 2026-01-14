# AFBMixer
## AFBMixer: Adaptive Frequency-Domain-Enhanced Multi-Branch Mixer is effective for Photovoltaic Power Forecasting
<p style="text-align:justify;">
  Photovoltaic (PV) power forecasting is critical for grid-connected dispatch, power absorption, and operational optimization of solar power plants. PV output exhibits strong randomness, intermittency, and multi-scale periodicity due to sunlight and meteorological factors. Current methods face challenges of imprecise multi-frequency modeling, inflexible band segmentation, and balancing computational complexity with accuracy. To address these, this study proposes Adaptive Frequency-Domain-Enhanced Multi-Branch Mixer (AFBMixer). First, the model’s DynamicFreqFeatureExtractor (DFFE) performs adaptive frequency band decomposition of PV data via learnable boundaries. By integrating multi-dimensional spectral features such as band energy distribution and spectral entropy, it accurately captures key frequency components including intraday illumination cycles and weather fluctuations. Furthermore, a scene-aware dynamic gating mechanism utilizes frequency-domain embeddings to adaptively modulate the amplitude of time-domain data, reinforcing valid signals while suppressing noise. To this end, a Frequency-Guided Multi-Branch Temporal Mixer adapts to multi-scale patterns. Crucially, the aggregated multi-dimensional spectral features are injected into the mixer to dynamically re-weight different MLP branches, achieving structural adaptation driven by spectral fingerprints. Employing an MLP-dominated architecture, AFBMixer balances computational efficiency with accuracy. Experimental results demonstrate the model's superior performance and efficiency in PV power forecasting.
</p>

---
# AFBMixer Model
<div align=center><img src="./AFBMixer.svg" alt="AFBMixer" width="800" height="850"></div>

---

***
（1）You can find the run scripts (model parameter files) for each dataset in the AFBMixer-Results/results-log folder. To maximize reproducibility, you can copy the parameters from these files to the corresponding locations in the run.py file.

（2）The models folder contains the implementation code files for the AFBMixer model.

（3）You need to modify the relevant paths in run.py and exp_data.py under the experiments folder according to your file paths (this is quite straightforward).

（4）Torch version is 2.0.0, Python version is 3.8. Required dependencies are listed in requirements.txt. The installation command is as follows (replace 'packname' with the desired package): {pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 'packname'}.












