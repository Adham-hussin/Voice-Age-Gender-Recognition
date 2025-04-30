# Project Notes and Resources

## Preprocessing Module
### Notes
- Steps:
    0. Format Normalization (No Need, All MP3)
    1. Resampling (No Need, All 48kHz)
    2. Denoising (LPF)
    3. Amplitude Normalization
    4. Tempo Normalization
    5. Handle Variable Length 
    6. Data Balancing (Handle Imbalanced Data)


### Resources
- [GfG - Preprocessing the Audio Dataset](https://www.geeksforgeeks.org/preprocessing-the-audio-dataset/)
- [GfG - Handling Imbalanced Data for Classification](https://www.geeksforgeeks.org/handling-imbalanced-data-for-classification/)

## Feature Extraction/Selection Module
### Notes
- Gender and age categories are called meta features (normally not the end goal of the project)
- table of features and their descriptions


| Feature | Description | How does it relate to the age-gender classification? |
| ------- | ----------- | ----------------------------------------------- |
| ZCR | Zero Crossing Rate - Measures the rate at which the signal changes from positive to negative or vice versa | High ZCR indicates a higher frequency of changes, often found in higher-pitched voices (younger individuals); lower ZCR correlates with deeper voices (typically older individuals). |
| MFCC1-39_mean/std | Mel-Frequency Cepstral Coefficients - Capture the spectral properties of the audio in a way that mimics human hearing. Each coefficient describes different frequency bands. The mean and standard deviation of each reflect central tendency and variation over time. | Age and gender affect vocal tract characteristics, which in turn influence MFCCs. For example, male voices tend to have lower frequency energy (lower MFCCs), while female or younger voices might have higher frequency emphasis. |
| Spectral Centroid | Indicates the center of mass of the spectrum, perceptually linked to the brightness of the sound. | Higher centroid typically indicates a brighter (higher-pitched) sound, associated with female and younger voices. Lower centroid suggests deeper, darker sounds often from males or older individuals. |
| Spectral Bandwidth | Measures the width of the spectrum - how much frequencies are spread out around the centroid. | Wider bandwidths can indicate breathier or more complex timbres, often associated with female or youthful voices; narrower bandwidths are common in more compact, deep voices. |
| Spectral Rolloff | The frequency below which a certain percentage (typically 85%) of the total spectral energy lies. | Higher rolloff points suggest more high-frequency content, typical in female or younger voices. Lower rolloffs are often associated with male or older voices. |
| Spectral Flatness | Describes how noise-like a sound is versus being tonal. Higher flatness = more noise-like. | Breathier or noisier voices, often linked to aging or some female speech patterns, may have higher flatness. Tonal voices (common in young males) show lower flatness. |
| RMS Energy | Root Mean Square Energy - a measure of the loudness of the signal. | Louder voices (higher RMS) can indicate vocal strength, sometimes associated with males or younger individuals. Softer voices may occur in older individuals. |
| Pitch Mean | Average fundamental frequency of the voice. | Female and younger voices generally have higher pitch, while male and older voices have lower pitch. |
| Pitch Std | Standard deviation of pitch - reflects how much the pitch varies. | High variation can indicate expressive or stressed speech, potentially correlated with emotional tone (affecting perception of age or gender). |
| Formant1 | First formant frequency - related to vowel height and vocal tract shape. | Lower in males due to longer vocal tracts; higher in females or children. |
| Formant2 | Second formant frequency - related to vowel backness and tongue position. | Helps differentiate gender and age due to articulatory differences. |
| Formant3 | Third formant frequency - further characterizes vocal tract resonance. | Subtle differences contribute to finer age-gender distinctions. |
| Jitter | Measures frequency variation between cycles. | Higher jitter is often associated with older or less stable voices. Lower jitter indicates smoother vocal fold vibrations, typical of younger or healthier speakers. |
| Shimmer | Measures amplitude variation between cycles. | Higher shimmer reflects less control or age-related voice degradation. Lower shimmer is common in younger, healthy voices. |
| HNR (Harmonics-to-Noise Ratio) | Ratio of harmonic sound to noise in the voice. | Higher HNR means a cleaner, more stable voice, often found in younger individuals. Lower HNR can indicate aging or vocal strain. |



### Resources
- [Voice Signals Features](https://maelfabien.github.io/machinelearning/Speech9/#4-zero-crossing-rate)
- [MFCC Article](http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)
- [Automatic speaker age and gender recognition using acoustic and prosodic level information fusion](https://www.sciencedirect.com/science/article/pii/S0885230812000101)
- [Voice-based gender and age recognition system](https://ieeexplore.ieee.org/abstract/document/10141801/)

## Model Selection and Training Module
### Notes
- grid search for model selection and hyperparameter tuning

### Resources
- 

## Performance Analysis Module
### Notes
- 

### Resources
- 