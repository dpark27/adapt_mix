#!/usr/bin/python

import os
import sys
import optparse
import numpy as np
import pandas as pd
import scipy.optimize as sp
from scipy.stats import mvn
from itertools import combinations


def main():
    parser = optparse.OptionParser()
    parser.add_option('-g', '--genotypes', dest='genotypes', action='store', type='string', default='')
    parser.add_option('-d', '--data_prefix', dest='data_prefix', action='store', type='string', default='')
    parser.add_option('-p', '--pop_list', dest='pop_list', action='store', type='string', default='CEU,YRI,GIH')
    parser.add_option('-c', '--chrom', dest='chrom', action='store', type='string', default='')
    parser.add_option('-z', '--zscores', dest='zscores', action='store', type='string', default='')
    parser.add_option('-l', '--use_likelihood_optimization', dest='use_likelihood_optimization', action='store', type='string', default='0')
    parser.add_option('-r', '--lambda_value', dest='lambda_value', action='store', type='string', default='0.1')
    parser.add_option('-w', '--window_size', dest='window_size', action='store', type='string', default='100')
    parser.add_option('-b', '--block_size', dest='block_size', action='store', type='string', default='100')
    parser.add_option('-o', '--output_file', dest='output_file', action='store', type='string', default='0')

    (options, args) = parser.parse_args()

    lambda_value = float(options.lambda_value)
    window_size = int(options.window_size)
    block_size = int(options.block_size)
    pop_list = options.pop_list.split(',')
    zscores = pd.read_pickle(options.zscores)
    # genotypes = pd.read_pickle(options.genotypes)

    weights = np.random.rand(len(pop_list))
    weights = weights / float(weights.sum())
    bounds = []
    for i in xrange(0, len(pop_list)):
        bounds.append((0, 1))

    o_file = open(options.output_file, 'wb')
    optimization_attempts = 0
    while optimization_attempts < 10:
        results = sp.fmin_l_bfgs_b(optimize_by_mse, approx_grad=True, x0=weights, args=(pop_list, options.data_prefix, options.chrom, zscores, lambda_value, window_size, block_size), bounds=bounds)
        optimized_weights = results[0]
        minimized_mse = results[1]
        optimization_info = results[2]

        if np.unique(optimized_weights).size > 1 or optimization_attempts == 9:
            o_file.write("Chrom: " + options.chrom + '\n')
            o_file.write("Pops: " + '\t'.join(pop_list) + '\n')
            o_file.write("Weights: " + '\t'.join(map(str, optimized_weights)) + '\n')
            o_file.write("MSE: " + str(minimized_mse) + '\n')
            break
        else:
            weights = np.random.rand(len(pop_list))
            weights = weights / float(weights.sum())

        optimization_attempts += 1
        # print optimization_info

    o_file.close()


######## optimization functions ############
def optimize_by_mse(weights, *args):
    # genotypes = args[0]
    pop_list = args[0] 
    data_prefix = args[1] 
    chrom = args[2] 
    zscores = args[3]
    lambda_value = args[4]
    window_size = args[5]
    block_size = args[5]

    mse = 1000
    squared_sum = 0.0
    imputed_count = 0
    count = 0
    block_idx = 0
    for i in xrange(0, zscores.shape[0], block_size):  # matrices are saved in blocks of 1000
        zscores_in_block = get_zscores_in_block(zscores, i, block_size)
        pop_means_in_block, pop_cors_in_block, pop_covs_in_block, pop_variances_in_block = load_pop_data(pop_list, data_prefix, block_idx, chrom)
        block_idx += 1
        window_idx = 0
        for j in xrange(0, zscores_in_block.shape[0], window_size):
            zscores_in_window = get_zscores_in_window(zscores_in_block, window_size, j)
            pop_means, pop_cors, pop_covs, pop_variances = get_pop_data_in_window(pop_means_in_block, pop_cors_in_block, pop_covs_in_block, pop_variances_in_block, window_size, j)
            window_idx += 1
            count += 1

            tmp_weights = weights / weights.sum()
            sigma = create_weighted_sigma(tmp_weights, pop_means, pop_cors, pop_variances, pop_covs, pop_list)
            all_good = filter_snps(pop_list, pop_means, pop_cors, sigma, zscores_in_window)
            if all_good.shape[0] == 0:
                continue

            good_zscores = zscores_in_window.iloc[all_good]
            good_sigma = sigma.iloc[all_good, all_good]
            good_sigma = good_sigma + np.identity(all_good.shape[0]) * lambda_value
            good_mafs = pop_means.iloc[all_good, :]

            for idx_to_impute in xrange(0, good_zscores.shape[0]):
                pop_mafs = good_mafs.iloc[idx_to_impute, :] / 2.0
                pop_mafs[pop_mafs > 0.5] = 1 - pop_mafs[pop_mafs > 0.5]
                weighted_maf = np.dot(tmp_weights, pop_mafs)
                if weighted_maf < 0.01:
                    continue

                try:
                    z_imputed = impute(good_zscores, good_sigma, idx_to_impute)
                except:
                    continue
                    
                if np.isnan(z_imputed):
                    continue

                imputed_count += 1

                # print window_idx, idx_to_impute, idx_to_impute, z_imputed, good_zscores.iloc[idx_to_impute]

                squared_sum += (z_imputed - good_zscores.iloc[idx_to_impute])**2

    if imputed_count > 0:
        mse = (squared_sum / imputed_count)[0]

    return mse


######### HELPER FUNCTIONS  ################
def dmvnorm(x, mean, sigma, log=False):
    dmvnorm = 0

    k = x.shape[0]
    part1 = np.exp(-0.5*k*np.log(2*np.pi))
    part2 = np.power(np.linalg.det(sigma), -0.5)
    dev = x.iloc[:, 0] - mean
    try:
        part3 = np.exp(-0.5*np.dot(np.dot(dev.transpose(),np.linalg.inv(sigma)),dev))
        dmvnorm = part1*part2*part3
        if log:
            dmvnorm = np.log(part1) + np.log(part2) + np.log(part3)
    except:
        dmvnorm = 0
        if log:
            dmvnorm = -np.inf

    return dmvnorm


def create_weighted_sigma(weights, pop_means, pop_cors, pop_variances, pop_covs, pop_list):
    # get weighted means
    weight_minus_one_adj_means = (weights - 1) * pop_means
    weight_adj_means = (weights) * pop_means

    # create adjustment factors for variances
    pop_ds = pd.DataFrame(index=pop_means.index, columns=pop_means.columns)
    for pop in pop_list:
        other_pops  = [p for p in weight_adj_means.columns if p not in [pop]]
        pop_ds[pop] = weight_minus_one_adj_means[pop] + weight_adj_means[other_pops].sum(axis=1)

    # create final variances
    final_variances = ((pop_ds**2) + pop_variances).apply(np.dot, axis=1, args=[weights])

    # create final covariances
    final_covariances = pd.DataFrame(index=pop_ds.index, columns=pop_ds.index)
    final_covariances = final_covariances.fillna(0)
    pop_idx = 0
    for pop in pop_list:
        tmp = weights[pop_idx]*(pop_covs[pop] + pd.DataFrame(np.outer(pop_ds[pop], pop_ds[pop])))
        final_covariances += tmp
        pop_idx += 1

    # create final correlations i.e. sigma
    denominators = pd.DataFrame(np.sqrt(np.outer(final_variances, final_variances)))
    final_sigma = final_covariances / denominators

    return final_sigma


def impute(zscores, sigma, idx_to_impute):
    window_size = zscores.shape[0]
    selections = create_selection_list(idx_to_impute, window_size)
    masked_z_scores = zscores.iloc[selections]

    sigma_obs = sigma.iloc[selections, selections]
    sigma_i = sigma.iloc[idx_to_impute, selections]

    weights = np.dot(sigma_i, np.linalg.inv(sigma_obs))
    denominator = np.sqrt(np.dot(np.dot(weights, sigma_obs), weights.T))

    z_imputed = np.dot(weights, masked_z_scores) / denominator

    return z_imputed


def create_selection_list(idx_to_impute, window_size):
    selections = []
    for i in xrange(0, window_size):
        if i != idx_to_impute:
            selections.append(i)
    return selections


######### FILTER CODE ########################
def filter_snps(pop_list, pop_means, pop_cors, sigma, zscores_in_window, filter_idx_to_impute=False):
    maf_good = filter_snps_by_maf(pop_means, zscores_in_window, filter_idx_to_impute)
    zscore_good = filter_snps_by_zscores(zscores_in_window)
    # cors_good = filter_snps_by_cors(pop_cors, pop_list, zscores_in_window)

    all_good = np.intersect1d(maf_good.tolist(), zscore_good.tolist())
    # all_good = np.intersect1d(t, cors_good.tolist())

    return all_good



def filter_snps_by_maf(pop_means, zscores_in_window, filter_idx_to_impute=False):
    pop_mafs = pop_means / 2
    filtered = pop_mafs[(pop_mafs >= 0.05) & (pop_mafs <= 0.95)].isnull().sum(axis=1)
    if not filter_idx_to_impute:
        filtered.iloc[zscores_in_window.shape[0]/2] = 0
    maf_good = np.where(filtered == 0)[0]

    return maf_good


def filter_snps_by_zscores(zscores_in_window):
    zscore_good = np.where(zscores_in_window.isnull() == False)[0]

    return zscore_good


def filter_snps_by_cors(pop_cors, pop_list, zscores_in_window):
    cors_good_intersect = np.array([])
    for pop in pop_list:
        cors = pop_cors[pop]
        cors_good = cors.iloc[:, zscores_in_window.shape[0]/2][pd.isnull(cors.iloc[:, zscores_in_window.shape[0]/2])==False].index
        if cors_good_intersect.shape[0] == 0:
            cors_good_intersect = cors_good
        else:
            cors_good_intersect = np.intersect1d(cors_good_intersect.tolist(), cors_good.tolist())

    return cors_good_intersect


def filter_snps_by_covs(pop_covs, pop_list, zscores_in_window):
    covs_good_intersect = None
    for pop in pop_list:
        covs = pop_covs[pop]
        covs_good = covs.iloc[:, zscores_in_window.shape[0]/2][pd.isnull(covs.iloc[:, zscores_in_window.shape[0]/2])==False].index
        if not covs_good_intersect:
            covs_good_intersect = covs_good
        else:
            covs_good_intersect = np.intersect1d(covs_good_intersect.tolist(), covs_good.tolist())

    return covs_good_intersect


######## DATA LOAD #########################
def load_pop_data(pop_list, data_prefix, block_idx, chrom):
    pop_means = load_pop_means(pop_list, data_prefix, block_idx, chrom)
    pop_cors = load_pop_cors(pop_list, data_prefix, block_idx, chrom)
    pop_covs = load_pop_covs(pop_list, data_prefix, block_idx, chrom)
    pop_variances = load_pop_variances(pop_list, data_prefix, block_idx, chrom)

    return pop_means, pop_cors, pop_covs, pop_variances


def load_pop_means(pop_list, data_prefix, block_idx, chrom):
    means = []
    for pop in pop_list:
        pop_means_file = data_prefix + pop.lower() + '/' + pop + '_means_chr' + chrom + '_' + str(block_idx) + '.pkl'
        means.append(pd.read_pickle(pop_means_file))

    pop_means = pd.concat(means, axis=1)
    pop_means.columns = pop_list

    return pop_means


def load_pop_cors(pop_list, data_prefix, block_idx, chrom):
    pop_cors = {}
    for pop in pop_list:
        pop_cors_file = data_prefix + pop.lower() + '/' + pop + '_cors_chr' + chrom + '_' + str(block_idx) + '.pkl'
        pop_cors[pop] = pd.read_pickle(pop_cors_file)

    return pop_cors


def load_pop_covs(pop_list, data_prefix, block_idx, chrom):
    pop_covs = {}
    for pop in pop_list:
        pop_covs_file = data_prefix + pop.lower() + '/' + pop + '_covs_chr' + chrom + '_' + str(block_idx) + '.pkl'
        pop_covs[pop] = pd.read_pickle(pop_covs_file)

    return pop_covs


def load_pop_variances(pop_list, data_prefix, block_idx, chrom):
    variances = []
    for pop in pop_list:
        pop_variances_file = data_prefix + pop.lower() + '/' + pop + '_variances_chr' + chrom + '_' + str(block_idx) + '.pkl'
        variances.append(pd.read_pickle(pop_variances_file))

    pop_variances = pd.concat(variances, axis=1)
    pop_variances.columns = pop_list

    return pop_variances


def get_zscores_in_block(zscores, i, block_size):
    zscores_in_block = zscores.iloc[i:(i+block_size)]
    zscores_in_block.index = range(0, zscores_in_block.shape[0])

    return zscores_in_block


#################### Window Code ################################
def get_pop_data_in_window(pop_means_in_block, pop_cors_in_block, pop_covs_in_block, pop_variances_in_block, window_size, j):
    pop_means_in_window = get_pop_means_in_window(pop_means_in_block, window_size, j)
    pop_cors_in_window = get_pop_cors_in_window(pop_cors_in_block, window_size, j)
    pop_covs_in_window = get_pop_covs_in_window(pop_covs_in_block, window_size, j)
    pop_variances_in_window = get_pop_variances_in_window(pop_variances_in_block, window_size, j)

    return pop_means_in_window, pop_cors_in_window, pop_covs_in_window, pop_variances_in_window


def get_zscores_in_window(zscores_in_block, window_size, j):
    zscores_in_window = zscores_in_block.iloc[j:(j+window_size)]
    zscores_in_window.index = range(0, zscores_in_window.shape[0])

    return zscores_in_window


def get_pop_means_in_window(pop_means_in_block, window_size, j):
    pop_means = pop_means_in_block.iloc[j:(j+window_size), :]
    pop_means.index = range(0, pop_means.shape[0])

    return pop_means


def get_pop_variances_in_window(pop_variances_in_block, window_size, j):
    pop_variances = pop_variances_in_block.iloc[j:(j+window_size), :]
    pop_variances.index = range(0, pop_variances.shape[0])

    return pop_variances


def get_pop_cors_in_window(pop_cors_in_block, window_size, j):
    pop_cors = {}
    for pop in pop_cors_in_block:
        pop_cors[pop] = pop_cors_in_block[pop].iloc[j:(j+window_size), j:(j+window_size)]
        pop_cors[pop].index = range(0, pop_cors[pop].shape[0])
        pop_cors[pop].columns = range(0, pop_cors[pop].shape[0])

    return pop_cors


def get_pop_covs_in_window(pop_covs_in_block, window_size, j):
    pop_covs = {}
    for pop in pop_covs_in_block:
        pop_covs[pop] = pop_covs_in_block[pop].iloc[j:(j+window_size), j:(j+window_size)]
        pop_covs[pop].index = range(0, pop_covs[pop].shape[0])
        pop_covs[pop].columns = range(0, pop_covs[pop].shape[0])

    return pop_covs


if __name__ == '__main__':
    main()

