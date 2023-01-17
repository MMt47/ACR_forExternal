# Automatic Chord Recognition Codes (for External)

This repo contains implementation of 2 models (as Baseline) for ACR task:  
**Bi-Directional Transformer** and **CRNN**(Convolutional Recurrent Neural Network).  

## Code Descriptions
- `A_utils/gen_features.py` : Generates features based on audio and chord data stored in "B_audiofiles", "B_chordfiles".
- `A_utils/gen_split_indices.py`: Splits Data for Train/Test and 5-Fold Cross Validation.
- `A_utils/torch_utils.py` : Defines classes and functions to be used when training and testing models.
- `D_BTCmodel/model.py` : Defines Bi-Directional Transformer model.
- `D_BTCmodel/modules.py` : Defines Modules for Bi-Directional Transformer model with Pytorch.
- `D_CRNNmodel/model.py` : Defines CRNN model with Pytorch.
- `train.py` : for training model.
- `test.py` : for testing model.
- `inference.py` : infers Chords from Audio File.



## Reference
[BTC]  
Jonggwon Park et al., "A Bi-Directional Transformer for Musical Chord Recognition", 20th ISMIR, pp. 620-627, 2019.

[CRNN]  
Junyan Jiang et al., "Large-vocabulary Chord Transcription Via Chord Structure Decomposition", 20th ISMIR, pp. 644-651, 2019.
