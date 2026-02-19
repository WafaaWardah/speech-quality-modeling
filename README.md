# Speech Quality Modeling  
### Transformer-Based Modeling for Speech Quality Prediction in Telecommunications

This repository accompanies the doctoral thesis on transformer-based foundation models for multidimensional speech quality prediction in telecommunications.

Speech quality is modeled as a perceptual construct comprising:

- Overall Quality (MOS)
- Noisiness
- Discontinuity
- Coloration
- Loudness

The repository is organized according to the main experimental chapters of the thesis.

---

## Chapter 4: Zero-Shot Evaluation

Evaluation of pretrained transformer-based speech foundation models to analyze whether their representations encode perceptually relevant speech quality information.

Models evaluated:

- wav2vec 2.0  
- HuBERT  
- WavLM  
- Whisper  
- Audio Spectrogram Transformer (AST)  

All encoders are kept frozen and evaluated using identical downstream regression mappings.

---

## Chapter 5: Multi-Task Modeling

Development of a joint prediction framework that simultaneously estimates:

- Overall speech quality  
- Noisiness  
- Discontinuity  
- Coloration  
- Loudness  

The Audio Spectrogram Transformer (AST) serves as the backbone model.

---

## Chapter 6: Single-Task Modeling

Dimension-specific modeling approach in which each perceptual attribute is learned independently.

This allows analysis of:

- Task-specific representation specialization  
- Performance trade-offs between joint and independent modeling  

The implementation for this part is available in:

https://github.com/WafaaWardah/SQ-AST

Please cite [this paper](https://www.isca-archive.org/interspeech_2025/wardah25_interspeech.html) when using this model.

Cite as: Wardah, W., Spang, R.P., Barriac, V., Reimes, J., Llagostera, A., Berger, J., Möller, S. (2025) SQ-AST: A Transformer-Based Model for Speech Quality Prediction. Proc. Interspeech 2025, 2335-2339, doi: 10.21437/Interspeech.2025-2683
---

## Chapter 7: Out-of-Domain Application

Application of dimension-specific models to synthetic speech quality and naturalness assessment.

The implementation for this part is available in:

https://github.com/WafaaWardah/Synth-AST

---

