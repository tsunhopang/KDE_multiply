import numpy as np
import scipy.stats
import warnings


def KDE_multiply(KDE1, KDE2, bw_method=None, downsample=False,
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

    The corresponding combined KDE is returned cotaining
    nsamples samples if ``nsamples'' is provided

    Parameters
    ----------
    KDE1, KDE2 : scipy.stats.gaussian_kde
        The Gaussian KDEs to be combined.
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth for the
        combined KDE. Please see
        `https://docs.scipy.org/doc/scipy/reference/'
        `generated/scipy.stats.gaussian_kde.html'
        for details
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
    if downsample and not random_state:
        warnings.warn("It is suggested to have random_state"
                      "provided for reproducibility")

    # calculate the covariance matrix for the combined Gaussians
    cov_joint = np.linalg.inv(KDE1.inv_cov + KDE2.inv_cov)

    # calculate the means for the combined Gaussians
    x1 = KDE1.dataset
    x2 = KDE2.dataset

    x1_contribution = np.linalg.multi_dot((cov_joint, KDE1.inv_cov, x1))
    x2_contribution = np.linalg.multi_dot((cov_joint, KDE2.inv_cov, x2))
    x3 = []
    for i in range(KDE1.d):
        x3_tmp = np.add.outer(x1_contribution[i], x2_contribution[i])
        x3_tmp = x3_tmp.flatten()
        x3.append(x3_tmp)
    x3 = np.array(x3)

    # calculate the weight for the combined samples
    w1 = KDE1._weights
    w2 = KDE2._weights
    w3 = np.multiply.outer(w1, w2)
    w3 = w3.flatten()
    w3 /= np.sum(w3)

    # downsample the resulting KDE otherwise it will take ages
    # to do a KDE.logpdf(x)
    if downsample:
        if not nsamples:
            nsamples = int(np.mean([KDE1.n, KDE2.n]))
        if isinstance(random_state, int):
            r = np.random.RandomState(random_state)
            x3 = r.choice(x3, size=nsamples)
        elif random_state:
            x3 = random_state.choice(x3, size=nsamples)
        else:
            x3 = np.random.choice(x3, size=nsamples)

    KDE3 = scipy.stats.gaussian_kde(x3, weights=w3, bw_method=bw_method)
    KDE3.covariance = cov_joint
    KDE3.inv_cov = KDE1.inv_cov + KDE2.inv_cov

    return KDE3
