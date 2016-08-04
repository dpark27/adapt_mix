#!/usr/bin/python

import os
import sys
import optparse
import numpy as np
import pandas as pd
from itertools import combinations

def main():
    parser = optparse.OptionParser()
    parser.add_option('-p', '--ped_file', dest='ped_file', action='store', type='string', default='')
    parser.add_option('-s', '--sample_file', dest='sample_file', action='store', type='string', default='')
    parser.add_option('-l', '--population', dest='population', action='store', type='string', default='MXL')
    parser.add_option('-o', '--output_dir', dest='output_dir', action='store', type='string', default='.')
    parser.add_option('-c', '--chrom', dest='chrom', action='store', type='string', default='.')
    parser.add_option('-b', '--block_size', dest='block_size', action='store', type='string', default='1000')

    (options, args) = parser.parse_args()

    block_size = int(options.block_size)
    population = options.population
    samples = load_samples(options.sample_file)
    pop_mask = samples['POP'] == population
    row_idx_in_pop = samples[pop_mask].index

    chrom = options.chrom
    pop_genotypes = load_genotypes(options.ped_file, row_idx_in_pop)
    pop_genotypes_pd = pd.DataFrame(pop_genotypes)
    pop_genotypes_file = options.output_dir + '/' + population + '_genos_chr' + chrom + '.pkl'
    # uncomment below to save the genotypes if you want
    # pop_genotypes_pd.to_pickle(pop_genotypes_file)

    count = 0
    for i in xrange(0, pop_genotypes.shape[1], block_size):
        pop_genotypes_i = pop_genotypes[:, i:(i+block_size)]
        pop_genotypes_i[np.where(pop_genotypes_i<0)] = np.nan
        pop_genotypes_i_pd = pd.DataFrame(pop_genotypes_i)

        correlation_matrix = pop_genotypes_i_pd.corr()
        pop_cors_file = options.output_dir + '/' + population + '_cors_chr' + chrom + '_' + str(count) + '.pkl'
        correlation_matrix.to_pickle(pop_cors_file)

        covariance_matrix = pop_genotypes_i_pd.cov()
        pop_covs_file = options.output_dir + '/' + population + '_covs_chr' + chrom + '_' + str(count) + '.pkl'
        covariance_matrix.to_pickle(pop_covs_file)

        means = pop_genotypes_i_pd.mean()
        pop_means_file = options.output_dir + '/' + population + '_means_chr' + chrom + '_' + str(count) + '.pkl'
        means.to_pickle(pop_means_file)

        variances = pop_genotypes_i_pd.var()
        pop_variances_file = options.output_dir + '/' + population + '_variances_chr' + chrom + '_' + str(count) + '.pkl'
        variances.to_pickle(pop_variances_file)

        count += 1


def load_samples(sample_file):
    samples = pd.read_csv(sample_file, delimiter='\t')

    return samples


def load_genotypes(ped_file, row_idx_in_pop):
    col_count = 0
    i_file = open(ped_file, 'rb')
    for line in i_file:
        if col_count == 0:
            col_count = len(line.strip().split()) - 6
            break
    i_file.close()

    genotypes = np.empty([row_idx_in_pop.shape[0], col_count/2])
    gt_idx = 0
    line_count = 0
    i_file = open(ped_file, 'rb')
    for line in i_file:
        if line_count not in row_idx_in_pop:
            line_count += 1
            continue
        else:
            haps = np.array(line.strip().split()[6:], dtype=np.int)
            genos = (haps[::2]-1) + (haps[1::2]-1)
            genotypes[gt_idx, ] = genos
            gt_idx += 1
            line_count += 1
    i_file.close()

    return genotypes


if __name__ == '__main__':
    main()
