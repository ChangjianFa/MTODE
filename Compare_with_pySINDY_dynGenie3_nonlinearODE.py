#coding=utf-8
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from itertools import combinations
from scipy.integrate import solve_ivp
import pandas as pd
from functools import partial
import time
import os


os.getcwd()
#n_repeat=20 # repet trils
np.random.seed(123) #Reproducibility
#snr = 1 #SNR, nosie level
cutoff = 0.1 #cutoff for Variable Importance(used in nonlinear_ODE and dynGenie3)
t = np.arange(0, 1.01, 0.01) #times
n = len(t) #total samples
dt = t[1] - t[0]


# 1. simulate data(Same as R script)
def sim20(snr,t=t,n=len(t),dt = t[1]-t[0]):

    def psi(x):
        return np.array([np.sin(0.5 * x), x ** 3])

    n_order = 2
    p = 20
    dim_feature = n_order * p
    thetas = np.zeros((dim_feature, p))

    thetas[3, 0] = 7
    thetas[1, 1] = -11
    thetas[7, 2] = 4
    thetas[5, 3] = -2
    thetas[11, 4] = -2
    thetas[9, 5] = 3

    thetas[15, 6] = 5
    thetas[13, 7] = -5
    thetas[13, 8] = 12
    thetas[12, 9] = 2
    thetas[13, 10] = -5
    thetas[12, 11] = 5
    thetas[13, 12] = -8
    thetas[12, 13] = -6

    thetas[31, 14] = -4
    thetas[29, 15] = 4
    thetas[34, 16] = -3
    thetas[33, 17] = 6
    thetas[39, 18] = -2
    thetas[37, 19] = 2

    thetas[10, 1] = 5
    thetas[7, 4] = -5
    thetas[37, 16] = -2
    thetas[29, 19] = -2

    thetas[10, 8] = -2
    thetas[0, 10] = 4
    thetas[34, 12] = 3
    thetas[31, 13] = -1

    # diagonal
    for i in range(p):
        thetas[2 * i, i] = -2

    true_adj = np.zeros((p, p))  # row is each function, columns regulate rows
    for i in range(p):
        x = thetas[:, i]  # 列向量
        groups = [x[j:j + 2] for j in range(0, len(x), 2)]  # 每2个元素一组

        # 计算每组的和是否非零
        idx = [k for k, g in enumerate(groups) if np.sum(g) != 0]

        true_adj[i, idx] = 1

    def mod1(t, state):
        state_exp = np.concatenate([psi(x) for x in state])
        dx = thetas.T @ state_exp
        return dx

    State0 = np.round(np.random.normal(0, 2, size=p), 1)

    sol = solve_ivp(fun=mod1, t_span=(t[0], t[-1]), y0=State0, t_eval=t, method='LSODA', rtol=1e-8, atol=1e-10)

    X = sol.y.T
    t = sol.t
    signal_var = np.var(X, axis=0)
    snr_power_ratio = 10 ** (snr / 10)
    noise_var = signal_var / snr_power_ratio
    noise_sd = np.sqrt(noise_var)
    noise = np.random.normal(scale=noise_sd, size=(n, p))
    X_obs = X + noise
    return [X,X_obs,t,true_adj]

X,X_obs,t,true_adj = sim20(snr)

# 2. functions for anlysis
def run_pySINDY(X,t):
    import pysindy as ps
    from sklearn.linear_model import Lasso
    library_functions = [lambda X: np.sin(0.5 * X), lambda X: X**3]
    lib = ps.CustomLibrary(library_functions=library_functions)
    lib = lib.fit(X)
    #lib.get_feature_names()
    model0 = ps.SINDy(feature_library=lib,optimizer=ps.STLSQ())
    model0.fit(X, t=t)

    library_functions = [lambda X: np.sin(0.5 * X), lambda X: X ** 3]
    lib = ps.CustomLibrary(library_functions=library_functions)
    lib = lib.fit(X)
    model1 = ps.SINDy(feature_library=lib,optimizer=ps.SR3())
    model1.fit(X, t=t)

    library_functions = [lambda X: np.sin(0.5 * X), lambda X: X ** 3]
    lib = ps.CustomLibrary(library_functions=library_functions)
    lib = lib.fit(X)
    model2 = ps.SINDy(feature_library=lib, optimizer=Lasso())
    model2.fit(X, t=t)


    def plot_fit(select=0):
        import matplotlib.pyplot as plt
        Xdot_true = model.differentiate(X, t)
        #select = 3
        fit = basis@coeffs.T
        dx0 = Xdot_true[:,select]
        fit0 = fit[:,select]
        # 5. MSE error
        mse = np.mean((fit0 - dx0)**2)
        print("MSE between fit0 and dx0:", mse)

        # 6. Plot
        plt.figure(figsize=(10,5))
        plt.plot(t, dx0, label="True dx/dt (dx1)")
        plt.plot(t, fit0, '--', label="SINDy fit (fit0)")
        plt.xlabel("time")
        plt.ylabel("dx10dt")
        plt.legend()
        plt.title("True derivative vs SINDy fit")
        plt.show()

    def acquire_adj(p,coeffs):
        adj = np.zeros((p, p))
        for i in range(p):
            coef = np.array(coeffs[i,:])
            n_total = len(coef)
            coef_2row = np.array([coef[0:p],coef[p:n_total]])
            idx = np.where(np.sum(coef_2row, axis=0) != 0)[0]
            adj[i, idx] = 1
        return adj

    adj0 = acquire_adj(p=20, coeffs=model0.coefficients())
    adj1 = acquire_adj(p=20, coeffs=model1.coefficients())
    adj2 = acquire_adj(p=20, coeffs=model2.coefficients())
    return [adj0, adj1, adj2]

def run_dynGenie3(X,t):
    #copy all code from https://github.com/vahuynh/dynGENIE3/blob/master/dynGENIE3_python/dynGENIE3.py
    from sklearn.tree import BaseDecisionTree
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    import numpy as np
    import time
    from operator import itemgetter
    from multiprocessing import Pool
    from itertools import combinations

    def compute_feature_importances(estimator):

        """Computes variable importances from a trained tree-based model.
        """

        if isinstance(estimator, BaseDecisionTree):
            return estimator.tree_.compute_feature_importances(normalize=False)
        else:
            importances = [e.tree_.compute_feature_importances(normalize=False)
                           for e in estimator.estimators_]
            importances = np.array(importances)
            return np.sum(importances, axis=0) / len(estimator)

    def get_link_list(VIM, gene_names=None, regulators='all', maxcount='all', file_name=None):

        """Gets the ranked list of (directed) regulatory links.

        Parameters
        ----------

        VIM: numpy array
            Array as returned by the function dynGENIE3(), in which the element (i,j) is the score of the edge directed from the i-th gene to the j-th gene.

        gene_names: list of strings, optional
            List of length p, where p is the number of rows/columns in VIM, containing the names of the genes. The i-th item of gene_names must correspond to the i-th row/column of VIM. When the gene names are not provided, the i-th gene is named Gi.
            default: None

        regulators: list of strings, optional
            List containing the names of the candidate regulators. When a list of regulators is provided, the names of all the genes must be provided (in gene_names), and the returned list contains only edges directed from the candidate regulators. When regulators is set to 'all', any gene can be a candidate regulator.
            default: 'all'

        maxcount: 'all' or positive integer, optional
            Writes only the first maxcount regulatory links of the ranked list. When maxcount is set to 'all', all the regulatory links are written.
            default: 'all'

        file_name: string, optional
            Writes the ranked list of regulatory links in the file file_name.
            default: None



        Returns
        -------

        The list of regulatory links, ordered according to the edge score. Auto-regulations do not appear in the list. Regulatory links with a score equal to zero are randomly permuted. In the ranked list of edges, each line has format:

            regulator   target gene     score of edge
        """

        # Check input arguments
        if not isinstance(VIM, np.ndarray):
            raise ValueError('VIM must be a square array')
        elif VIM.shape[0] != VIM.shape[1]:
            raise ValueError('VIM must be a square array')

        ngenes = VIM.shape[0]

        if gene_names is not None:
            if not isinstance(gene_names, (list, tuple)):
                raise ValueError('input argument gene_names must be a list of gene names')
            elif len(gene_names) != ngenes:
                raise ValueError(
                    'input argument gene_names must be a list of length p, where p is the number of columns/genes in the expression data')

        if regulators != 'all':
            if not isinstance(regulators, (list, tuple)):
                raise ValueError('input argument regulators must be a list of gene names')

            if gene_names is None:
                raise ValueError('the gene names must be specified (in input argument gene_names)')
            else:
                sIntersection = set(gene_names).intersection(set(regulators))
                if not sIntersection:
                    raise ValueError('The genes must contain at least one candidate regulator')

        if maxcount != 'all' and not isinstance(maxcount, int):
            raise ValueError('input argument maxcount must be "all" or a positive integer')

        if file_name is not None and not isinstance(file_name, str):
            raise ValueError('input argument file_name must be a string')

        # Get the indices of the candidate regulators
        if regulators == 'all':
            input_idx = list(range(ngenes))
        else:
            input_idx = [i for i, gene in enumerate(gene_names) if gene in regulators]

        nTFs = len(input_idx)

        # Get the non-ranked list of regulatory links
        vInter = [(i, j, score) for (i, j), score in np.ndenumerate(VIM) if i in input_idx and i != j]

        # Rank the list according to the weights of the edges
        vInter_sort = sorted(vInter, key=itemgetter(2), reverse=True)
        nInter = len(vInter_sort)

        # Random permutation of edges with score equal to 0
        flag = 1
        i = 0
        while flag and i < nInter:
            (TF_idx, target_idx, score) = vInter_sort[i]
            if score == 0:
                flag = 0
            else:
                i += 1

        if not flag:
            items_perm = vInter_sort[i:]
            items_perm = np.random.permutation(items_perm)
            vInter_sort[i:] = items_perm

        # Write the ranked list of edges
        nToWrite = nInter
        if isinstance(maxcount, int) and maxcount >= 0 and maxcount < nInter:
            nToWrite = maxcount

        if file_name:

            outfile = open(file_name, 'w')

            if gene_names is not None:
                for i in range(nToWrite):
                    (TF_idx, target_idx, score) = vInter_sort[i]
                    TF_idx = int(TF_idx)
                    target_idx = int(target_idx)
                    outfile.write('%s\t%s\t%.6f\n' % (gene_names[TF_idx], gene_names[target_idx], score))
            else:
                for i in range(nToWrite):
                    (TF_idx, target_idx, score) = vInter_sort[i]
                    TF_idx = int(TF_idx)
                    target_idx = int(target_idx)
                    outfile.write('G%d\tG%d\t%.6f\n' % (TF_idx + 1, target_idx + 1, score))

            outfile.close()

        else:

            if gene_names is not None:
                for i in range(nToWrite):
                    (TF_idx, target_idx, score) = vInter_sort[i]
                    TF_idx = int(TF_idx)
                    target_idx = int(target_idx)
                    print('%s\t%s\t%.6f' % (gene_names[TF_idx], gene_names[target_idx], score))
            else:
                for i in range(nToWrite):
                    (TF_idx, target_idx, score) = vInter_sort[i]
                    TF_idx = int(TF_idx)
                    target_idx = int(target_idx)
                    print('G%d\tG%d\t%.6f' % (TF_idx + 1, target_idx + 1, score))

    def estimate_degradation_rates(TS_data, time_points):

        """
        For each gene, the degradation rate is estimated by assuming that the gene expression x(t) follows:
        x(t) =  A exp(-alpha * t) + C_min,
        between the highest and lowest expression values.
        C_min is set to the minimum expression value over all genes and all samples.
        """

        ngenes = TS_data[0].shape[1]
        nexp = len(TS_data)

        C_min = TS_data[0].min()
        if nexp > 1:
            for current_timeseries in TS_data[1:]:
                C_min = min(C_min, current_timeseries.min())

        alphas = np.zeros((nexp, ngenes))

        for (i, current_timeseries) in enumerate(TS_data):
            current_time_points = time_points[i]

            for j in range(ngenes):

                idx_min = np.argmin(current_timeseries[:, j])
                idx_max = np.argmax(current_timeseries[:, j])

                xmin = current_timeseries[idx_min, j]
                xmax = current_timeseries[idx_max, j]

                if xmin != xmax:
                    tmin = current_time_points[idx_min]
                    tmax = current_time_points[idx_max]

                    xmin = max(xmin - C_min, 1e-6)
                    xmax = max(xmax - C_min, 1e-6)

                    xmin = np.log(xmin)
                    xmax = np.log(xmax)

                    alphas[i, j] = (xmax - xmin) / abs(tmin - tmax)

        alphas = alphas.max(axis=0)

        return alphas

    def dynGENIE3(TS_data, time_points, alpha='from_data', SS_data=None, gene_names=None, regulators='all',
                  tree_method='RF', K='sqrt', ntrees=1000, compute_quality_scores=False, save_models=False, nthreads=1):

        '''Computation of tree-based scores for all putative regulatory links.

        Parameters
        ----------

        TS_data: list of numpy arrays
            List of arrays, where each array contains the gene expression values of one time series experiment. Each row of an array corresponds to a time point and each column corresponds to a gene. The i-th column of each array must correspond to the same gene.

        time_points: list of one-dimensional numpy arrays
            List of n vectors, where n is the number of time series (i.e. the number of arrays in TS_data), containing the time points of the different time series. The i-th vector specifies the time points of the i-th time series of TS_data.

        alpha: either 'from_data', a positive number or a vector of positive numbers
            Specifies the degradation rate of the different gene expressions.
            When alpha = 'from_data', the degradation rate of each gene is estimated from the data, by assuming an exponential decay between the highest and lowest observed expression values.
            When alpha is a vector of positive numbers, the i-th element of the vector must specify the degradation rate of the i-th gene.
            When alpha is a positive number, all the genes are assumed to have the same degradation rate alpha.
            default: 'from_data'

        SS_data: numpy array, optional
            Array containing steady-state gene expression values. Each row corresponds to a steady-state condition and each column corresponds to a gene. The i-th column/gene must correspond to the i-th column/gene of each array of TS_data.
            default: None

        gene_names: list of strings, optional
            List of length p containing the names of the genes, where p is the number of columns/genes in each array of TS_data. The i-th item of gene_names must correspond to the i-th column of each array of TS_data (and the i-th column of SS_data when SS_data is not None).
            default: None

        regulators: list of strings, optional
            List containing the names of the candidate regulators. When a list of regulators is provided, the names of all the genes must be provided (in gene_names). When regulators is set to 'all', any gene can be a candidate regulator.
            default: 'all'

        tree-method: 'RF' or 'ET', optional
            Specifies which tree-based procedure is used: either Random Forest ('RF') or Extra-Trees ('ET')
            default: 'RF'

        K: 'sqrt', 'all' or a positive integer, optional
            Specifies the number of selected attributes at each node of one tree: either the square root of the number of candidate regulators ('sqrt'), the number of candidate regulators ('all'), or any positive integer.
            default: 'sqrt'

        ntrees: positive integer, optional
            Specifies the number of trees grown in an ensemble.
            default: 1000

        compute_quality_scores: boolean, optional
            Indicates if the scores assessing the edge ranking quality must be computed or not. These scores are:
            - the score of prediction of out-of-bag samples, i.e. the Pearson correlation between the predicted and true output values. To be able to compute this score, Random Forests must be used (i.e. parameter tree_method must be set to 'RF').
            - the stability score, measuring the average stability among the top-5 candidate regulators returned by each tree of a forest.
            default: False

        save_models: boolean, optional
            Indicates if the tree models (one for each gene) must be saved or not.

        nthreads: positive integer, optional
            Number of threads used for parallel computing
            default: 1


        Returns
        -------

        A tuple (VIM, alphas, prediction_score, stability_score, treeEstimators).

        VIM: array in which the element (i,j) is the score of the edge directed from the i-th gene to the j-th gene. All diagonal elements are set to zero (auto-regulations are not considered). When a list of candidate regulators is provided, all the edges directed from a gene that is not a candidate regulator are set to zero.

        alphas: vector in which the i-th element is the degradation rate of the i-th gene.

        prediction_score: prediction score on out-of-bag samples (averaged over all genes and all trees). prediction_score is an empty list if compute_quality_scores is set to False or if tree_method is not set to 'RF'.

        stability_score: stability score (averaged over all genes). stability_score is an empty list if compute_quality_scores is set to False.

        treeEstimators: list of tree models, where the i-th model is the model predicting the expression of the i-th gene. treeEstimators is an empty list if save_models is set to False.

        '''

        time_start = time.time()

        # Check input arguments
        if not isinstance(TS_data, (list, tuple)):
            raise ValueError(
                'TS_data must be a list of arrays, where each row of an array corresponds to a time point/sample and each column corresponds to a gene')

        for expr_data in TS_data:
            if not isinstance(expr_data, np.ndarray):
                raise ValueError(
                    'TS_data must be a list of arrays, where each row of an array corresponds to a time point/sample and each column corresponds to a gene')

        ngenes = TS_data[0].shape[1]

        if len(TS_data) > 1:
            for expr_data in TS_data[1:]:
                if expr_data.shape[1] != ngenes:
                    raise ValueError('The number of columns/genes must be the same in every array of TS_data.')

        if not isinstance(time_points, (list, tuple)):
            raise ValueError(
                'time_points must be a list of n one-dimensional arrays, where n is the number of time series experiments in TS_data')

        if len(time_points) != len(TS_data):
            raise ValueError(
                'time_points must be a list of n one-dimensional arrays, where n is the number of time series experiments in TS_data')

        for tp in time_points:
            if (not isinstance(tp, (list, tuple, np.ndarray))) or (isinstance(tp, np.ndarray) and tp.ndim > 1):
                raise ValueError(
                    'time_points must be a list of n one-dimensional arrays, where n is the number of time series in TS_data')

        for (i, expr_data) in enumerate(TS_data):
            if len(time_points[i]) != expr_data.shape[0]:
                raise ValueError(
                    'The length of the i-th vector of time_points must be equal to the number of rows in the i-th array of TS_data')

        if alpha != 'from_data':
            if not isinstance(alpha, (list, tuple, np.ndarray, int, float)):
                raise ValueError(
                    "input argument alpha must be either 'from_data', a positive number or a vector of positive numbers")

            if isinstance(alpha, (int, float)) and alpha < 0:
                raise ValueError("the degradation rate(s) specified in input argument alpha must be positive")

            if isinstance(alpha, (list, tuple, np.ndarray)):
                if isinstance(alpha, np.ndarray) and alpha.ndim > 1:
                    raise ValueError(
                        "input argument alpha must be either 'from_data', a positive number or a vector of positive numbers")
                if len(alpha) != ngenes:
                    raise ValueError(
                        'when input argument alpha is a vector, this must be a vector of length p, where p is the number of genes')
                for a in alpha:
                    if a < 0:
                        raise ValueError("the degradation rate(s) specified in input argument alpha must be positive")

        if SS_data is not None:
            if not isinstance(SS_data, np.ndarray):
                raise ValueError(
                    'SS_data must be an array in which each row corresponds to a steady-state condition/sample and each column corresponds to a gene')

            if SS_data.ndim != 2:
                raise ValueError(
                    'SS_data must be an array in which each row corresponds to a steady-state condition/sample and each column corresponds to a gene')

            if SS_data.shape[1] != ngenes:
                raise ValueError(
                    'The number of columns/genes in SS_data must by the same as the number of columns/genes in every array of TS_data.')

        if gene_names is not None:
            if not isinstance(gene_names, (list, tuple)):
                raise ValueError('input argument gene_names must be a list of gene names')
            elif len(gene_names) != ngenes:
                raise ValueError(
                    'input argument gene_names must be a list of length p, where p is the number of columns/genes in the expression data')

        if regulators != 'all':
            if not isinstance(regulators, (list, tuple)):
                raise ValueError('input argument regulators must be a list of gene names')

            if gene_names is None:
                raise ValueError('the gene names must be specified (in input argument gene_names)')
            else:
                sIntersection = set(gene_names).intersection(set(regulators))
                if not sIntersection:
                    raise ValueError('The genes must contain at least one candidate regulator')

        if tree_method != 'RF' and tree_method != 'ET':
            raise ValueError('input argument tree_method must be "RF" (Random Forests) or "ET" (Extra-Trees)')

        if K != 'sqrt' and K != 'all' and not isinstance(K, int):
            raise ValueError('input argument K must be "sqrt", "all" or a stricly positive integer')

        if isinstance(K, int) and K <= 0:
            raise ValueError('input argument K must be "sqrt", "all" or a stricly positive integer')

        if not isinstance(ntrees, int):
            raise ValueError('input argument ntrees must be a stricly positive integer')
        elif ntrees <= 0:
            raise ValueError('input argument ntrees must be a stricly positive integer')

        if not isinstance(compute_quality_scores, bool):
            raise ValueError('input argument compute_quality_scores must be a boolean (True or False)')

        if not isinstance(save_models, bool):
            raise ValueError('input argument save_models must be a boolean (True or False)')

        if not isinstance(nthreads, int):
            raise ValueError('input argument nthreads must be a stricly positive integer')
        elif nthreads <= 0:
            raise ValueError('input argument nthreads must be a stricly positive integer')

        # Re-order time points in increasing order
        for (i, tp) in enumerate(time_points):
            tp = np.array(tp, dtype=np.float32)
            indices = np.argsort(tp)
            time_points[i] = tp[indices]
            expr_data = TS_data[i]
            TS_data[i] = expr_data[indices, :]

        if alpha == 'from_data':
            alphas = estimate_degradation_rates(TS_data, time_points)
        elif isinstance(alpha, (int, float)):
            alphas = np.zeros(ngenes) + float(alpha)
        else:
            alphas = [float(a) for a in alpha]

        print('Tree method: ' + str(tree_method))
        print('K: ' + str(K))
        print('Number of trees: ' + str(ntrees))
        print('alpha min: ' + str(min(alphas)))
        print('alpha max: ' + str(max(alphas)))
        print('\n')

        # Get the indices of the candidate regulators
        if regulators == 'all':
            input_idx = list(range(ngenes))
        else:
            input_idx = [i for i, gene in enumerate(gene_names) if gene in regulators]

        # Learn an ensemble of trees for each target gene and compute scores for candidate regulators
        VIM = np.zeros((ngenes, ngenes))

        if compute_quality_scores:
            if tree_method == 'RF':
                prediction_score = np.zeros(ngenes)
            else:
                prediction_score = []
            stability_score = np.zeros(ngenes)
        else:
            prediction_score = []
            stability_score = []

        if save_models:
            treeEstimators = [0] * ngenes
        else:
            treeEstimators = []

        if nthreads > 1:
            print('running jobs on %d threads' % nthreads)

            input_data = [
                [TS_data, time_points, SS_data, i, alphas[i], input_idx, tree_method, K, ntrees, compute_quality_scores,
                 save_models] for i in range(ngenes)]

            pool = Pool(nthreads)
            alloutput = pool.map(wr_dynGENIE3_single, input_data)

            for out in alloutput:
                i = out[0]

                (vi, prediction_score_i, stability_score_i, treeEstimator) = out[1]
                VIM[i, :] = vi

                if compute_quality_scores:
                    if tree_method == 'RF':
                        prediction_score[i] = prediction_score_i
                    stability_score[i] = stability_score_i

                if save_models:
                    treeEstimators[i] = treeEstimator

        else:
            print('running single threaded jobs')
            for i in range(ngenes):
                print('Gene %d/%d...' % (i + 1, ngenes))

                (vi, prediction_score_i, stability_score_i, treeEstimator) = dynGENIE3_single(TS_data, time_points,
                                                                                              SS_data, i, alphas[i],
                                                                                              input_idx, tree_method, K,
                                                                                              ntrees,
                                                                                              compute_quality_scores,
                                                                                              save_models)
                VIM[i, :] = vi

                if compute_quality_scores:
                    if tree_method == 'RF':
                        prediction_score[i] = prediction_score_i
                    stability_score[i] = stability_score_i

                if save_models:
                    treeEstimators[i] = treeEstimator

        VIM = np.transpose(VIM)
        if compute_quality_scores:
            if tree_method == 'RF':
                prediction_score = np.mean(prediction_score)
            stability_score = np.mean(stability_score)

        time_end = time.time()
        print("Elapsed time: %.2f seconds" % (time_end - time_start))

        return VIM, alphas, prediction_score, stability_score, treeEstimators

    def wr_dynGENIE3_single(args):
        return ([args[3],
                 dynGENIE3_single(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8],
                                  args[9], args[10])])

    def dynGENIE3_single(TS_data, time_points, SS_data, output_idx, alpha, input_idx, tree_method, K, ntrees,
                         compute_quality_scores, save_models):

        h = 1  # lag (in number of time points) used for the finite approximation of the derivative of the target gene expression
        ntop = 5  # number of top-ranked candidate regulators over which to compute the stability score

        ngenes = TS_data[0].shape[1]
        nexp = len(TS_data)
        nsamples_time = sum([expr_data.shape[0] for expr_data in TS_data])
        ninputs = len(input_idx)

        # Construct learning sample

        # Time-series data
        input_matrix_time = np.zeros((nsamples_time - h * nexp, ninputs))
        output_vect_time = np.zeros(nsamples_time - h * nexp)

        # Data for the computation of the prediction score on out-of-bag samples
        output_vect_time_present = np.zeros(nsamples_time - h * nexp)
        output_vect_time_future = np.zeros(nsamples_time - h * nexp)
        time_diff = np.zeros(nsamples_time - h * nexp)

        nsamples_count = 0

        for (i, current_timeseries) in enumerate(TS_data):
            current_time_points = time_points[i]
            npoints = current_timeseries.shape[0]
            time_diff_current = current_time_points[h:] - current_time_points[:npoints - h]
            current_timeseries_input = current_timeseries[:npoints - h, input_idx]
            current_timeseries_output = (current_timeseries[h:, output_idx] - current_timeseries[:npoints - h,
                                                                              output_idx]) / time_diff_current + alpha * current_timeseries[
                                                                                                                         :npoints - h,
                                                                                                                         output_idx]
            nsamples_current = current_timeseries_input.shape[0]
            input_matrix_time[nsamples_count:nsamples_count + nsamples_current, :] = current_timeseries_input
            output_vect_time[nsamples_count:nsamples_count + nsamples_current] = current_timeseries_output
            output_vect_time_present[nsamples_count:nsamples_count + nsamples_current] = current_timeseries[
                                                                                         :npoints - h, output_idx]
            output_vect_time_future[nsamples_count:nsamples_count + nsamples_current] = current_timeseries[h:,
                                                                                        output_idx]
            time_diff[nsamples_count:nsamples_count + nsamples_current] = time_diff_current
            nsamples_count += nsamples_current

        # Steady-state data (if any)
        if SS_data is not None:

            input_matrix_steady = SS_data[:, input_idx]
            output_vect_steady = SS_data[:, output_idx] * alpha

            # Concatenation
            input_all = np.vstack([input_matrix_steady, input_matrix_time])
            output_all = np.concatenate((output_vect_steady, output_vect_time))

            del input_matrix_time
            del output_vect_time
            del input_matrix_steady
            del output_vect_steady

        else:

            input_all = input_matrix_time
            output_all = output_vect_time

            del input_matrix_time
            del output_vect_time

        # Parameters of the tree-based method

        # Whether or not to compute the prediction score of out-of-bag samples
        if compute_quality_scores and tree_method == 'RF':
            oob_score = True
        else:
            oob_score = False

        # Parameter K of the tree-based method
        if (K == 'all') or (isinstance(K, int) and K >= len(input_idx)):
            max_features = "auto"
        else:
            max_features = K

        if tree_method == 'RF':
            treeEstimator = RandomForestRegressor(n_estimators=ntrees, max_features=max_features, oob_score=oob_score)
        elif tree_method == 'ET':
            treeEstimator = ExtraTreesRegressor(n_estimators=ntrees, max_features=max_features, oob_score=oob_score)

        # Learn ensemble of trees
        treeEstimator.fit(input_all, output_all)

        # Compute importance scores
        feature_importances = compute_feature_importances(treeEstimator)
        vi = np.zeros(ngenes)
        vi[input_idx] = feature_importances
        vi[output_idx] = 0

        # Normalize importance scores
        vi_sum = np.sum(vi)
        if vi_sum > 0:
            vi = vi / vi_sum

        # Ranking quality scores
        prediction_score_oob = []
        stability_score = []

        if compute_quality_scores:

            if tree_method == 'RF':

                # Prediction of out-of-bag samples

                if SS_data is not None:

                    nsamples_SS = SS_data.shape[0]

                    # Samples coming from the steady-state data
                    oob_prediction_SS = treeEstimator.oob_prediction_[:nsamples_SS]
                    output_pred_SS = oob_prediction_SS / alpha

                    # Samples coming from the time series data
                    oob_prediction_TS = treeEstimator.oob_prediction_[nsamples_SS:]
                    output_pred_TS = (
                                                 oob_prediction_TS - alpha * output_vect_time_present) * time_diff + output_vect_time_present

                    output_pred = np.concatenate((output_pred_SS, output_pred_TS))
                    output_true = np.concatenate((SS_data[:, output_idx], output_vect_time_future))

                    (prediction_score_oob, tmp) = np.pearsonr(output_pred, output_true)

                else:
                    oob_prediction_TS = treeEstimator.oob_prediction_
                    output_pred_TS = (
                                                 oob_prediction_TS - alpha * output_vect_time_present) * time_diff + output_vect_time_present

                    (prediction_score_oob, tmp) = np.pearsonr(output_pred_TS, output_vect_time_future)

            # Stability score

            # Importances returned by each tree
            importances_by_tree = np.array(
                [e.tree_.compute_feature_importances(normalize=False) for e in treeEstimator.estimators_])
            if output_idx in input_idx:
                idx = input_idx.index(output_idx)
                # Remove importances of target gene
                importances_by_tree = np.delete(importances_by_tree, idx, 1)

            # Add some jitter to avoir numerical errors
            importances_by_tree = importances_by_tree + np.random.uniform(low=1e-12, high=1e-11,
                                                                          size=importances_by_tree.shape)

            if np.sum(importances_by_tree) > 0:

                # Ranking of candidate regulators
                ranking_by_tree = [importances_by_tree[i, :].argsort()[::-1] for i in range(ntrees)]
                top_by_tree = [set(r[:ntop]) for r in ranking_by_tree]

                # Stability score computed over the top-ranked candidate regulators
                stability_score = np.mean([len(top_by_tree[i].intersection(top_by_tree[j])) for (i, j) in
                                           combinations(range(ntrees), 2)]) / float(ntop)


            # Variance of output is too small --> no forest was built and all the importances are zero
            else:
                stability_score = 0.0

        if save_models:
            return vi, prediction_score_oob, stability_score, treeEstimator
        else:
            return vi, prediction_score_oob, stability_score, []

    def dynGENIE3_predict_doubleKO(expr_WT, treeEstimators, alpha, gene_names, regulators, KO1_gene, KO2_gene,
                                   nTimePoints, deltaT):

        '''Prediction of gene expressions in a double knockout experiment.

        Parameters
        ----------

        expr_WT: vector containing the gene expressions in the wild-type.

        treeEstimators: list of tree models, as returned by the function dynGENIE3(), where the i-th model is the model predicting the expression of the i-th gene.
            The i-th model must correspond to the i-th gene in expr_WT.

        alpha: a positive number or a vector of positive numbers
            Specifies the degradation rate of the different gene expressions.
            When alpha is a vector of positives, the i-th element of the vector must specify the degradation rate of the i-th gene.
            When alpha is a positive number, all the genes are assumed to have the same degradation rate.

        gene_names: list of strings
            List containing the names of the genes. The i-th item of gene_names must correspond to the i-th gene in expr_WT.

        regulators: list of strings
            List containing the names of the candidate regulators. When regulators is set to 'all', any gene can be a candidate regulator.
            Note that the candidate regulators must be the same as the ones used when calling the function dynGENIE3().

        KO1_gene: name of the first knocked-out gene.

        KO2_gene: name of the second knocked-out gene.

        nTimePoints: integer
            Specifies the number of time points for which to make a prediction.

        deltaT: a positive number
            Specifies the (constant) time interval between two predictions.



        Returns
        -------

        An array in which the element (t,i) is the predicted expression of the i-th gene at the t-th time point.
        The first row of the array contains the initial gene expressions (i.e. the expressions in expr_WT), where the expressions of the two knocked-out genes are set to 0.

        '''

        time_start = time.time()

        # Check input arguments
        if not isinstance(expr_WT, np.ndarray) or expr_WT.ndim > 1:
            raise ValueError("input argument expr_WT must be a vector of numbers")

        ngenes = len(expr_WT)

        if len(treeEstimators) != ngenes:
            raise ValueError(
                "input argument treeEstimators must contain p tree models, where p is the number of genes in expr_WT")

        if not isinstance(alpha, (list, tuple, np.ndarray, int, float)):
            raise ValueError("input argument alpha must be a positive number or a vector of positive numbers")

        if isinstance(alpha, (int, float)) and alpha < 0:
            raise ValueError("the degradation rate(s) specified in input argument alpha must be positive")

        if isinstance(alpha, (list, tuple, np.ndarray)):
            if isinstance(alpha, np.ndarray) and alpha.ndim > 1:
                raise ValueError("input argument alpha must be a positive number or a vector of positive numbers")
            if len(alpha) != ngenes:
                raise ValueError(
                    'when input argument alpha is a vector, this must be a vector of length p, where p is the number of genes')
            for a in alpha:
                if a < 0:
                    raise ValueError("the degradation rate(s) specified in input argument alpha must be positive")

        if not isinstance(gene_names, (list, tuple)):
            raise ValueError('input argument gene_names must be a list of gene names')
        elif len(gene_names) != ngenes:
            raise ValueError(
                'input argument gene_names must be a list of length p, where p is the number of genes in expr_WT')

        if regulators != 'all':
            if not isinstance(regulators, (list, tuple)):
                raise ValueError('input argument regulators must be a list of gene names')

            sIntersection = set(gene_names).intersection(set(regulators))
            if not sIntersection:
                raise ValueError('The genes must contain at least one candidate regulator')

        if not (KO1_gene in gene_names):
            raise ValueError('input argument KO1_gene was not found in gene_names')

        if not (KO2_gene in gene_names):
            raise ValueError('input argument KO2_gene was not found in gene_names')

        if not isinstance(nTimePoints, int) or nTimePoints < 1:
            raise ValueError("input argument nTimePoints must be a strictly positive integer")

        if not isinstance(deltaT, (int, float)) or deltaT < 0:
            raise ValueError("input argument deltaT must be a positive number")

        KO1_idx = gene_names.index(KO1_gene)
        KO2_idx = gene_names.index(KO2_gene)

        geneidx = list(range(ngenes))
        geneidx.remove(KO1_idx)
        geneidx.remove(KO2_idx)

        if isinstance(alpha, (int, float)):
            alphas = np.zeros(ngenes) + float(alpha)
        else:
            alphas = [float(a) for a in alpha]

        # Get the indices of the candidate regulators
        if regulators == 'all':
            input_idx = list(range(ngenes))
        else:
            input_idx = [i for i, gene in enumerate(gene_names) if gene in regulators]

        # Predict time series

        print('Predicting time series...')

        TS_predict = np.zeros((nTimePoints + 1, ngenes))
        TS_predict[0, :] = expr_WT
        TS_predict[0, KO1_idx] = 0
        TS_predict[0, KO2_idx] = 0

        for t in range(1, nTimePoints + 1):
            new_expr = [(treeEstimators[i].predict(TS_predict[t - 1, input_idx].reshape(1, -1)) - alphas[i] *
                         TS_predict[t - 1, i]) * deltaT + TS_predict[t - 1, i] for i in geneidx]
            TS_predict[t, geneidx] = np.array(new_expr, dtype=np.float32).flatten()

        time_end = time.time()
        print("Elapsed time: %.2f seconds" % (time_end - time_start))

        return TS_predict


    #begin analysis
    # try author provided data
    # with open('TS_data.pkl', 'rb') as f:
    #    (TS_data, time_points, decay_rates, gene_names) = pickle.load(f)

    TS_data = list([X, X]) #do not use multiple source, rep X data
    time_points = list([t, t]) #same as above
    (VIM, alphas, prediction_score, stability_score, treeEstimators) = dynGENIE3(TS_data, time_points)
    # get_link_list(VIM)
    # The first column shows the regulator, the second column shows the target gene, and the last column
    # indicates the score of the link.

    return VIM.T

def run_nonliner_ODE(X,t):
    #code from https://github.com/lab319/GRNs_nonlinear_ODEs/blob/master/xgbgrn.py
    from xgboost import XGBRegressor

    def get_links(VIM, gene_names, regulators, sort=True, file_name=None):
        idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
        pred_edges = [(gene_names[j], gene_names[i], score) for (i, j), score in np.ndenumerate(VIM) if
                      i != j and j in idx]
        pred_edges = pd.DataFrame(pred_edges)
        if sort is True:
            pred_edges.sort_values(2, ascending=False, inplace=True)
        if file_name is None:
            print(pred_edges)
        else:
            pred_edges.to_csv(file_name, sep='\t', header=None, index=None)

    def estimate_degradation_rates(TS_data, time_points):
        """
        For each gene, the degradation rate is estimated by assuming that the gene expression x(t) follows:
        x(t) =  A exp(-alpha * t) + C_min,
        between the highest and lowest expression values.
        C_min is set to the minimum expression value over all genes and all samples.
        The function is available at the study named dynGENIE3.
        Huynh-Thu, V., Geurts, P. dynGENIE3: dynamical GENIE3 for the inference of
        gene networks from time series expression data. Sci Rep 8, 3384 (2018) doi:10.1038/s41598-018-21715-0
        """

        ngenes = TS_data[0].shape[1]
        nexp = len(TS_data)

        C_min = TS_data[0].min()
        if nexp > 1:
            for current_timeseries in TS_data[1:]:
                C_min = min(C_min, current_timeseries.min())

        alphas = np.zeros((nexp, ngenes))

        for (i, current_timeseries) in enumerate(TS_data):
            current_time_points = time_points[i]

            for j in range(ngenes):
                idx_min = np.argmin(current_timeseries[:, j])
                idx_max = np.argmax(current_timeseries[:, j])

                xmin = current_timeseries[idx_min, j]
                xmax = current_timeseries[idx_max, j]

                tmin = current_time_points[idx_min]
                tmax = current_time_points[idx_max]

                xmin = max(xmin - C_min, 1e-6)
                xmax = max(xmax - C_min, 1e-6)

                xmin = np.log(xmin)
                xmax = np.log(xmax)

                alphas[i, j] = (xmax - xmin) / abs(tmin - tmax)

        alphas = alphas.max(axis=0)

        return alphas

    def get_importances(TS_data, time_points, alpha="from_data",
                        SS_data=None, gene_names=None,
                        regulators='all', param={}):
        time_start = time.time()

        ngenes = TS_data[0].shape[1]

        if alpha is "from_data":
            alphas = estimate_degradation_rates(TS_data, time_points)
        else:
            alphas = [alpha] * ngenes

        # Get the indices of the candidate regulators
        idx = [i for i, gene in enumerate(gene_names) if gene in regulators]

        # Learn an ensemble of trees for each target gene, and compute scores for candidate regulators
        VIM = np.zeros((ngenes, ngenes))

        for i in range(ngenes):
            input_idx = idx.copy()
            if i in input_idx:
                input_idx.remove(i)
            vi = get_importances_single(TS_data, time_points, alphas[i], input_idx, i, SS_data, param)
            VIM[i, :] = vi

        time_end = time.time()
        print("Elapsed time: %.2f seconds" % (time_end - time_start))

        return VIM

    def get_importances_single(TS_data, time_points, alpha, input_idx, output_idx, SS_data, param):
        h = 1  # define the value of time step

        ngenes = TS_data[0].shape[1]
        nexp = len(TS_data)
        nsamples_time = sum([expr_data.shape[0] for expr_data in TS_data])
        ninputs = len(input_idx)

        # Construct training sample
        # Time-series data
        input_matrix_time = np.zeros((nsamples_time - h * nexp, ninputs))
        output_vect_time = np.zeros(nsamples_time - h * nexp)

        nsamples_count = 0
        for (i, current_timeseries) in enumerate(TS_data):
            current_time_points = time_points[i]
            npoints = current_timeseries.shape[0]
            time_diff_current = current_time_points[h:] - current_time_points[:npoints - h]
            current_timeseries_input = current_timeseries[:npoints - h, input_idx]
            current_timeseries_output = (current_timeseries[h:, output_idx] - current_timeseries[:npoints - h,
                                                                              output_idx]) / time_diff_current + alpha * current_timeseries[
                                                                                                                         :npoints - h,
                                                                                                                         output_idx]
            nsamples_current = current_timeseries_input.shape[0]
            input_matrix_time[nsamples_count:nsamples_count + nsamples_current, :] = current_timeseries_input
            output_vect_time[nsamples_count:nsamples_count + nsamples_current] = current_timeseries_output
            nsamples_count += nsamples_current

        # Steady-state data
        if SS_data is not None:
            input_matrix_steady = SS_data[:, input_idx]
            output_vect_steady = SS_data[:, output_idx] * alpha

            # Concatenation
            input_all = np.vstack([input_matrix_steady, input_matrix_time])
            output_all = np.concatenate((output_vect_steady, output_vect_time))
        else:
            input_all = input_matrix_time
            output_all = output_vect_time

        treeEstimator = XGBRegressor(**param)

        # Learn ensemble of trees
        treeEstimator.fit(input_all, output_all)

        # Compute importance scores
        feature_importances = treeEstimator.feature_importances_
        vi = np.zeros(ngenes)
        vi[input_idx] = feature_importances

        return vi

    def get_scores(VIM, gold_edges, gene_names, regulators):
        idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
        pred_edges = [(gene_names[j], gene_names[i], score) for (i, j), score in np.ndenumerate(VIM) if
                      i != j and j in idx]
        pred_edges = pd.DataFrame(pred_edges)
        # Take the top 100,000 predicated results
        pred_edges = pred_edges.iloc[:100000]
        final = pd.merge(pred_edges, gold_edges, on=[0, 1], how='inner')
        auroc = roc_auc_score(final['2_y'], final['2_x'])
        aupr = average_precision_score(final['2_y'], final['2_x'])

        return auroc, aupr

    #use args in originl code
    xgb_kwargs = dict(n_estimators=398, learning_rate=0.0133, importance_type="weight", max_depth=5, n_jobs=-1)

    """
    #test data provided by original author
    TS_data = pd.read_csv("insilico_size10_1_timeseries.tsv", sep='\t').values
    gold_edges = pd.read_csv("insilico_size10_1_goldstandard.tsv", sep= '\t', header=None)
    i = np.arange(0, 85, 21)
    j = np.arange(21, 106, 21)
    # get the time-series data
    TS_data = [TS_data[i:j] for (i, j) in zip(i, j)]
    time_points = [np.arange(0, 1001, 50)] * 5
    ngenes = TS_data[0].shape[1]
    gene_names = ['G'+str(i+1) for i in range(ngenes)]
    regulators = gene_names.copy()
    auroc, aupr = get_scores(VIM, gold_edges, gene_names, regulators)
    """
    #begin analysis
    ngenes = X.shape[1]
    gene_names = ['G' + str(i + 1) for i in range(ngenes)]
    regulators = gene_names.copy()
    TS_data = list([X, X])
    time_points = list([t, t])
    VIM = get_importances(TS_data, time_points, gene_names=gene_names,
                          regulators=regulators, param=xgb_kwargs)

    return VIM

def eval_adj(true_adj, pred_adj):
    from sklearn.metrics import matthews_corrcoef
    """
    true_adj: (p,p) ground truth adjacency matrix
    pred_adj: (p,p) predicted adjacency matrix (binary or continuous)
    """
    # 二值化（若 pred_adj 是连续系数，例如 SINDy 输出）
    pred_bin = (pred_adj != 0).astype(int)
    true_bin = (true_adj != 0).astype(int)

    # 展开成向量
    y_true = true_bin.flatten()
    y_pred = pred_bin.flatten()

    # 计算四种类型
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    # MCC
    MCC = matthews_corrcoef(y_true, y_pred)

    return [TP, FP, TN, FN, MCC]



# 3. analysis results
"""

pySINDY_adj0,pySINDY_adj1,pySINDY_adj2 = run_pySINDY(X_obs,t)
nonliner_ODE_adj = run_nonliner_ODE(X_obs,t)
nonliner_ODE_adj = np.where(nonliner_ODE_adj >= cutoff, 1, 0)
dynGenie3_adj = run_dynGenie3(X_obs,t)
dynGenie3_adj = np.where(dynGenie3_adj >= cutoff, 1, 0)
res = eval_adj(true_adj, dynGenie3_adj)
print(res)

"""

def run_once(seed, snr=10,cutoff=0.1):

    np.random.seed(seed)
    # simulate data
    X, X_obs, t, true_adj = sim20(snr)

    # run pySINDY
    adj0, adj1, adj2 = run_pySINDY(X_obs, t)

    # run dynGENIE3
    adj3 = run_dynGenie3(X_obs, t)
    adj3 = np.where(adj3  >= cutoff, 1, 0)
    # run nonlionerODE
    adj4 = run_nonliner_ODE(X_obs, t)
    adj4 = np.where(adj4  >= cutoff, 1, 0)

    # degree_true = np.sum(true_adj != 0, axis=0) + np.sum(true_adj != 0, axis=1)
    # degree_est = np.sum(adj != 0, axis=0) + np.sum(adj != 0, axis=1)
    # degree_est

    # evaluate
    results = {}
    for name, adj in zip(
        ["SINDY_STLSQ", "SINDY_SR3", "SINDY_Lasso", "dynGENIE3","nonlinear_ODE"],
        [adj0, adj1, adj2, adj3, adj4]
    ):
        TP, FP, TN, FN, MCC = eval_adj(true_adj,adj)
        results[name] = dict(
            TP=TP, FP=FP, TN=TN, FN=FN, MCC=MCC, seed=seed
        )

    return results

#results = run_once(seed=1000)

def run_n_times(n_repeat=10,snr=10):
    all_rows = []

    for i in range(n_repeat):
        seed = 1000 + i
        out = run_once(seed,snr = snr)

        for method, vals in out.items():
            vals['method'] = method
            all_rows.append(vals)

        print(f"Finished round {i+1}/{n_repeat}, seed={seed}")

    df = pd.DataFrame(all_rows)
    return df


"""
def run_n_times_parallel(n_repeat=10,snr=None, n_jobs=None):
    if n_jobs is None:
        n_jobs = max(cpu_count() - 1, 1)   # 留一个 CPU 给系统

    seeds = [1000 + i for i in range(n_repeat)]

    print(f"Using {n_jobs} cores to run {n_repeat} simulations...")

    with Pool(n_jobs) as pool:
        outs = pool.map(run_once, seeds)

    all_rows = []
    for seed, out in zip(seeds, outs):
        for method, vals in out.items():
            vals['method'] = method
            vals['seed'] = seed
            all_rows.append(vals)

    return pd.DataFrame(all_rows)
"""

def run_n_times_parallel(n_repeat=10, snr=None, n_jobs=None):
    if n_jobs is None:
        n_jobs = max(cpu_count() - 1, 1)

    seeds = [1000 + i for i in range(n_repeat)]
    print(f"Using {n_jobs} cores to run {n_repeat} simulations with SNR={snr}...")

    func = partial(run_once, snr=snr)

    with Pool(n_jobs) as pool:
        outs = pool.map(func, seeds)

    all_rows = []
    for seed, out in zip(seeds, outs):
        for method, vals in out.items():
            vals['method'] = method
            vals['seed'] = seed
            vals['SNR'] = snr
            all_rows.append(vals)

    return pd.DataFrame(all_rows)

def eval_method(df_res, snr_value):
    methods = pd.unique(df_res['method'])
    rows = []
    for m in methods:
        subset = df_res[df_res['method'] == m]

        TPR = subset["TP"] / (subset["TP"] + subset["FN"])
        FPR = subset["FP"] / (subset["FP"] + subset["TN"])
        MCC = subset["MCC"]

        row = {
            "method": m,
            "mean_TPR": TPR.mean(),
            "sd_TPR": TPR.std(),
            "mean_FPR": FPR.mean(),
            "sd_FPR": FPR.std(),
            "mean_MCC": MCC.mean(),
            "sd_MCC": MCC.std(),
            "SNR": snr_value
        }
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    return df_summary


def run_for_snr_range(snr_start=1, snr_end=2, n_repeat=2, n_jobs=None):
    df_list = []
    for snr in range(snr_start, snr_end + 1):
        print(f"Running SNR = {snr}")

        df_res = run_n_times_parallel(n_repeat=n_repeat, snr=snr, n_jobs=n_jobs)
        #df_res = run_n_times(n_repeat=n_repeat, snr=snr)
        df_summary = eval_method(df_res, snr_value=snr)
        df_list.append(df_summary)
        df_all = pd.concat(df_list, ignore_index=True)

    return df_all




if __name__ == "__main__":
    """
    #run once
    df_res = run_n_times_parallel(n_repeat=50, n_jobs=30)
    filenme = "20P_"+"SNR"+str(snr)+"_Cutoff"+str(cutoff) +"_out.csv"
    df_res.to_csv(filenme)
    eval_method(df_res,snr_value=2)
    print(f"Saved {filenme}")
    """
    filenme = "20P_" + "_Cutoff" + str(cutoff) + "_out.csv"
    df_all = run_for_snr_range(snr_start=1, snr_end=20, n_repeat=50, n_jobs=20)
    df_all.to_csv(filenme)
    print(f"Saved {filenme}")




