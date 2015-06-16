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


# save_matrices_by_pop.py
This script saves the cov/var/mean/cor matrices for a given reference population.  

Options:
-p/--ped_file: PLINK PED file to calculate the matrices from.  Expected to be in recode12 format, 1 chromosome per PED file
-s/--sample_file: Tab-delimited file that lists the samples in the same order as they are in the PED file and the corresponding reference population they belong to.
    ex.
    SAMPLEID POP
    idv1  CEU
    idv2  YRI
    idv3  MXL
    
-l/--population: Population you wish to compute the matrices for. Default: 'MXL'
-o/--output_dir: Directory to save the matices
-c/--chrom: Chromosome the PED file corresponds to. ex. 1 
-b/--block_size: Size of windows (in # of SNPs) to split the chromosome into. Default: 1000
