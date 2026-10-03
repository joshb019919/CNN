# CNN
A simple CNN for the Computer Vision class at Missouri State.

## AI Use Statement

I utilized Gemini Notebook to assist me with understanding this assignment beyond what I'd already learned.  I used it to help me understand traceback output, too, so I could more easily debug problems.

## Source Statements

### Model Pipeline

I built the model from course material with a little help from Gemini to debug some issues, like making sure to pass images directly to the CNN class instead of assigning it to the model attribute.  This ensures that Torch's `__call__()` wrapper is involved and all appropriate training hooks are employed or explicitly setting it to train mode before looping through the epochs.

### Train Pipeline

This pipeline is directly from the course's material.

### Test Pipeline

The test pipeline is directly from [Sling Academy](https://www.slingacademy.com/article/step-by-step-guide-to-pytorch-model-testing/#1.-setting-up-your-environment).
