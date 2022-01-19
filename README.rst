============
KDE_multiply
============

A function to multiply two scipy Gaussian KDEs analytically with arbitary dimension.

Usage
-----
.. code:: python

    from KDE_multiply import KDE_multiply
    from scipy.stats import gaussian_kde

    KDE1 = gaussian_kde(x1) # x1 is some generic one- or multi-dimensional samples
    KDE2 = gaussian_kde(x2) # x2 is some generic one- or multi-dimensional samples

    KDE3 = KDE_multiply(KDE1, KDE2)

Example
-------
.. code:: python

    from KDE_multiply import KDE_multiply
    from numpy import dot
    from numpy.random import seed, uniform, multivariate_normal
    from numpy.linalg import multi_dot, inv
    from scipy.stats import gaussian_kde
    from matplotlib import pyplot as plt
    from corner import corner

    # generate samples to be used for KDE
    dimension = 4 # setting the dimension
    seed(42) # setting the seed

    # randomly generate the mean for the two Gaussians
    mean1 = uniform(-3, 3, size=dimension)
    mean2 = uniform(-3, 3, size=dimension)

    # randomly generate the covariance matrix for the two Gaussians
    cov1 = uniform(0, 2, size=(dimension,dimension))
    cov1 = dot(cov1.T, cov1)
    cov2 = uniform(0, 2, size=(dimension,dimension))
    cov2 = dot(cov2.T, cov2)

    # generate samples for the two Gaussians
    x1 = multivariate_normal(mean=mean1, cov=cov1, size=6000).T
    x2 = multivariate_normal(mean=mean2, cov=cov2, size=6000).T

    # estimated the KDEs 
    KDE1 = gaussian_kde(x1)
    KDE2 = gaussian_kde(x2)

    # multiply the KDEs
    KDE_joint = KDE_multiply(KDE1, KDE2, downsample=True,
                             random_state=42, nsamples=6000)

    # resample from the joint KDE
    samples_joint = KDE_joint.resample(size=6000)

    # compare with exact calculation
    cov_predict = multi_dot((cov1, inv(cov1 + cov2), cov2))
    mean_predict = multi_dot((cov2, inv(cov1 + cov2), mean1))
    mean_predict += multi_dot((cov1, inv(cov1 + cov2), mean2)) 
    samples_joint_predict = multivariate_normal(mean=mean_predict,
                                                cov=cov_predict,
                                                size=6000).T

    fig = corner(samples_joint_predict.T, color='C0')
    corner(samples_joint.T, fig=fig, color='C1')
    plt.show()

.. image:: https://github.com/tsunhopang/KDE_multiply/blob/main/example.svg

Methods
-------
A Gaussian KDE is a weighted sum of Gaussians centered at the data points $\{\mu_i\}$ and having the same covariance matrix $\Sigma$ estimated based on the samples. Mathematically, it can be written as
.. math::
    {\\rm KDE} = \\sum_{i=0}^{N-1} w_i\\mathcal{N}(\\mu_i, \\Sigma),
where :math:`\{w_i\}` are the weights for each sample.\\

The main concept to be noted is that the product of two Gaussians is also a Gaussian. In particular,
.. math::
    \\mathcal{N}(\\mu_1, \\Sigma_1) \\times \\mathcal{N}(\\mu_2, \\Sigma_2) \\propto \\mathcal{N}(\\mu_3, \\Sigma_3),
where
.. math::
    \\Sigma_3 = \\Sigma_1 (\\Sigma_1 + \\Sigma_2)^{-1} \\Sigma_2,\\
.. math::
    \\mu_3 = \\Sigma_2 (\\Sigma_1 + \\Sigma_2)^{-1} \\mu_1 + \\Sigma_1 (\\Sigma_1 + \\Sigma_2)^{-1} \\mu_2.
As a result, the product of two Gaussian KDEs can be computed as
.. math::
    \begin{aligned}
            &{\\rm KDE}_1 \\times {\\rm KDE}_2 \\
            &= \\sum_{i=0}^{N-1} w_i\\mathcal{N}(\\mu_i, \\Sigma_1) \\times \\sum_{j=0}^{M-1} w_j\\mathcal{N}(\\mu_j, \\Sigma_2)\\
            &=\\sum_{i=0}^{N-1}\\sum_{j=0}^{M-1}w_iw_j\\mathcal{N}(\\mu_i, \\Sigma_1)\\mathcal{N}(\\mu_j, \\Sigma_2)\\
            &=\\sum_{k=0}^{MN-1} w_k \\mathcal{N}(\\mu_k, \\Sigma_3)\\
            &\\equiv {\\rm KDE}_3
    \end{aligned}
where
.. math::
    \begin{aligned}
    \\Sigma_3 &= \\Sigma_1 (\\Sigma_1 + \\Sigma_2)^{-1} \\Sigma_2,\\
    \\mu_k &= \\Sigma_2 (\\Sigma_1 + \\Sigma_2)^{-1} \\mu_i + \\Sigma_1 (\\Sigma_1 + \\Sigma_2)^{-1} \\mu_j\\
    w_k &= w_i \\times w_j
    \end{aligned}
\end{equation}
with :math:`k = N j + i`.
