# Testing
------------
## Test Order
------------
### order=0 
- no dependencies
- download ephys data example
- download BEHAVE_PKL
- download wavs

### order=1
- generate ACCURACIES_PKL
- generate PSYCHOMETRIC_PKL

### order=2
- generate num_shuffle_pkl(8)
- generate POP_PAIR_PKL
- generate psych_shuffle_pkl(1024)

### order=3
- generate ks_df(8)
- generate DERIVATIVE_PKL