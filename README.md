# detectron2 implementation of wally

Dataset: [https://github.com/tadejmagajna/HereIsWally](https://github.com/tadejmagajna/HereIsWally)

## usage
```bash

# setup the environment
./setup.sh gpu # gpu or cpu
source venv/bin/activate

# train
./train.py

# find waldo
./find.py <image>

# quit with "q"-key

# or run streamlit
streamlit run app.py
```
