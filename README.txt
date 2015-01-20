# optimize.py
This script optimizes weights over a given chromosome.
The population data files are expected to be pandas dataframe where genotypes are coded 0,1,2 
The z-scores are expected to be a 1xn pandas dataframe

Options:
-d/--data_prefix: prefix to where the reference panel files reside is ex. ~/refpanel_data/
-p/--pop_list: This is the list of populations in the reference panel ex. CEU,YRI
-c/--chrom:  This is the chromosome the weights are being optimized over; ex. 1
-z/--zscores:  The path the to the zscore files
-r/--lambda-value:  Lambda value for imputation, default is 0.1
-w/--window_size:  Window size for imputation, default is 100
-b/--block_size:  Number of SNPs in each reference panel data file, default 1000
-o/--output_file:  The path to output the weights
