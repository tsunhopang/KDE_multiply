import numpy as np
import scipy.stats
import scipy.special
import warnings


def KDE_multiply(KDE1, KDE2, downsample=False,
                 random_state=None, nsamples=None):
    """ Multiply two Gaussian KDEs analytically and return another
        Gaussian KDE

    As a Gaussian kernel density estimation is a sum of Gaussians with
    their means at the input dataset and sharing the same covariance matrix,
    multiplication of two of such kernel density estimation results
    in another Gaussian kernel density estimation with

    1) Joint covariance matrix = np.linalg.inv(KDE1.inv_cov + KDE2.inv_cov)
    2) The joint means
           = np.linalg.multi_dot((cov_joint, KDE1.inv_cov, x1))
           + np.linalg.multi_dot((cov_joint, KDE2.inv_cov, x2))
    3) The joint weights = np.multiply.outer((w1, w2)).flatten()
                           * normalization factor correction

    See section 8.1.8 in http://compbio.fmph.uniba.sk/vyuka/ml/old/2008/handouts/matrix-cookbook.pdf

    The corresponding combined KDE is returned cotaining
    nsamples samples if ``nsamples'' is provided

    Parameters
    ----------
    KDE1, KDE2 : scipy.stats.gaussian_kde
        The Gaussian KDEs to be combined.
    downsample : boolean, optional
        To downsample the samples within the combined KDE if set to True
    random_state : numpy.random.RandomState, integer, optional
        The numpy.random.RandomState or seed (if integer is provided)
        for the downsampling
    nsamples : int, optional
        The numpy of samples to be taken when downsample is True
        (Default : int(np.mean([KDE1.n, KDE2.n])

    Returns
    -------
    KDE3 : scipy.stats.gaussian_kde
        Resulting joint Gaussian KDE

    """

    # sanity checking
    if nsamples:
        assert nsamples < KDE1.n * KDE2.n, \
            "nsamples should be less than KDE1.n * KDE2.n"
    if not nsamples:
        nsamples = int(np.mean([KDE1.n, KDE2.n]))
        warnings.warn("No nsamples provided, setting to"
                      " int(np.mean([KDE1.n, KDE2.n]))")
    if downsample and not random_state:
        warnings.warn("It is suggested to have random_state"
                      " provided for reproducibility")

    # calculate the covariance matrix for the combined Gaussians
    cov_joint = np.linalg.inv(KDE1.inv_cov + KDE2.inv_cov)
    # also calculate the covariance matrix of the two KDEs
    cov1 = np.linalg.inv(KDE1.inv_cov)
    cov2 = np.linalg.inv(KDE2.inv_cov)

    # fetch the data from the two input KDEs
    x1 = KDE1.dataset
    x2 = KDE2.dataset

    w1 = KDE1._weights
    w2 = KDE2._weights

    # check if the array too big to fit in the memory
    try:
        x3_trial = []
        for i in range(KDE1.d):
            x3_trial.append(np.add.outer(x1[0], x2[0]))
        downsample_early = False
        del x3_trial
    except MemoryError:
        assert downsample, "Array too big, downsample is required"
        warnings.warn("The outer product array for the two datasets"
                      " is too big to fit into the memory, downsampling"
                      " at earlier stage now. This could result in"
                      " the joint dataset smaller than nsamples")
        downsample_early = True

    if downsample_early:
        nsamples_sqrt = int(np.sqrt(nsamples))
        if x1.shape[1] < nsamples_sqrt:
            nsamples_1 = x1.shape[1]
            nsamples_2 = nsamples // x1.shape[1]
        elif x2.shape[1] > nsamples_sqrt:
            nsamples_1 = nsamples // x2.shape[1]
            nsamples_2 = x2.shape[1]
        else:
            nsamples_1 = nsamples_sqrt
            nsamples_2 = nsamples_sqrt

        index_1 = index_random_choice(random_state, x1.shape[1], nsamples_1)
        index_2 = index_random_choice(random_state, x2.shape[1], nsamples_2)

        x1 = x1[:, index_1]
        x2 = x2[:, index_2]
        w1 = w1[index_1]
        w2 = w2[index_2]

    # calculate the means for the combined Gaussians
    x1_contribution = np.linalg.multi_dot((cov_joint, KDE1.inv_cov, x1))
    x2_contribution = np.linalg.multi_dot((cov_joint, KDE2.inv_cov, x2))
    x3 = []
    for i in range(KDE1.d):
        x3_tmp = np.add.outer(x1_contribution[i], x2_contribution[i])
        x3_tmp = x3_tmp.flatten()
        x3.append(x3_tmp)
    x3 = np.array(x3)

    # calculate the weight for the combined samples
    log_w3 = np.add.outer(np.log(w1), np.log(w2))
    log_w3 = log_w3.flatten()
    # also include the correction for the individual normalization
    # introduced in the Gaussian KDE
    cov_sum = cov1 + cov2
    x1_diff_x2 = []
    for i in range(KDE1.d):
        x1_diff_x2_tmp = np.subtract.outer(x1[i], x2[i])
        x1_diff_x2_tmp = x1_diff_x2_tmp.flatten()
        x1_diff_x2.append(x1_diff_x2_tmp)
    x1_diff_x2 = np.array(x1_diff_x2)
    expoential = np.einsum('ai,ab,bi->i',
                           x1_diff_x2,
                           np.linalg.inv(cov_sum),
                           x1_diff_x2)
    log_w3 += -0.5 * expoential
    log_w3 -= np.amin(log_w3)

    w3 = np.exp(log_w3)
    w3 /= np.sum(w3)

    # downsample the resulting KDE otherwise it will take ages
    # to do a KDE.logpdf(x)
    if downsample and not downsample_early:
        index = index_random_choice(random_state, x3.shape[1], nsamples, w3)
        x3 = x3[:, index]
        w3 = w3[index]

    KDE3 = scipy.stats.gaussian_kde(x3, weights=w3)
    KDE3.covariance = cov_joint
    KDE3.inv_cov = KDE1.inv_cov + KDE2.inv_cov

    return KDE3


def index_random_choice(random_state, length, N, p):

    if isinstance(random_state, int):
        r = np.random.RandomState(random_state)
        index = r.choice(length, size=N, p=p)
    elif random_state:
        index = random_state.choice(length, size=N, p=p)
    else:
        index = np.random.choice(length, size=N, p=p)

    return index
