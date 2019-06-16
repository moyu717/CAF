## Recommendation via Collaborative AutoregressiveFiltering (CAF)

This is the python implementation -- a item recommendation model using Collaborative Autoregressive Flows.

## Requirements
- python2.7
- tensorflow >= 1.3.0
- numpy >= 1.14.0

------------------

## Datasets
Here are three datasets we used in our paper. 
- CiteULike: <http://www.citeulike.org/>

- MovieLens: <https://grouplens.org/datasets/movielens/1M/>

- LastFM: <http://www.lastfm.com/>

| Datasets        |    #Users    | #Items  |#interaction |Sparsity |
| ------------- |:-------------:| :-----:|:--------:|:--------:|
| CiteULike     |5,551|  16,980|   204,986 |     99.8%|
| MovieLens     |6,040 | 3,544 |   993,482 |     95.4%|
| LastFM  |1,892  |17,632 |  92,834 |      97.3%|


For each dataset, we randomly select 70% of the user accessrecord as the training set and the remaining as the validation (10%)and testing (20%) data.

------------------

## Usage
To run CAF, please first clone the code (http://github.com/moyu717/CAF) to your python IDE(eg: Pycharm), then run the code CAF.py,  all the core codes are written in the file -- CAF.py.  After running CAF.py,  you may get a directory containing a caf_pmf.mat file.  Then you can copy caf_pmf.mat into test_matrics.py file to calculate Precision, Recall, Map, nDCG. 

To customize the code:

a、You can set this latent dimension (latent_size) to 100 or other dimensions(in our paper, we use 100),  which will affect the training time.  The larger the dimension, the longer the training time will be.  But the bigger the dimension,  the better the effect is not necessarily,  you need to find a suitable dimension.

b、You need to set the number of flows,  in our paper,  the best performance is achieved when K=7 for MovieLens and K=5 for CiteULike and LastFM on all metrics.

c、You can use your own data sets in the source code.
