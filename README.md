# Tools for training Automatic Speech Recognition models for Romanian

This repository consists of a set of modules that can be used for:  
  generating Romanian ASR training data based on Mozilla's Common Voice Corpus' Spanish and Italian sets  
  fine-tuning Whisper models using said data  
  benchmarking different models  
  
To install and run the tools it is necessary to run the following commands:   
`    git clone ...  `
`     git lfs install # if not already installed  `  
`     git lfs pull  `  
`     pip install -r requirements.txt  `  
`     chmod 777 ./setup.sh  `  
`     ./setup.sh  `  
