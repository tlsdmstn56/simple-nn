# simplenn

## Setup development environment

```bash
virtualenv venv
. venv/bin/activate

# simplenn deps
# simplenn has minimal dependencies, only numpy!
pip install numpy 

# simplenn example deps
pip install tqdm matplotlib  

# simplenn test deps
pip install pytest torch
```

## Usage

TBD

## Run

```bash
python ./example/train_linear_model.py
```

## Test

```bash
pytest
```

## TODO:

- Move forward/backward functionality to `Function` class and records call graph
    - Layer class is a just wrapper of collection of functions
- Implement call graph: networkx based?
- More layer: Conv, pooling, Dropout
    - Goal is to train/eval of resnet32
