# [RE] VAE with a VampPrior

We reimplement the methods and neural network architectures described in "VAE with a VampPrior" (Jakub M. Tomczak et al.). We briefly introduce and motivate variational autoencoders and the paper. To evaluate and compare the effects of different priors, we implement the standard 1-layered variational autoencoder architecture with a standard Gaussian prior and the VampPrior. Furthermore, we extend the VampPrior using the hierarchical, 2-layered architecture as described in the paper. We test both architectures with both priors on three datasets: MNIST, Omniglot, and Caltech101 (silhouettes). The results successfully re-verify the empirical findings of the original paper, showing the effectiveness of the VampPrior and the suggested hierarchical architecture.

## Authors

Alphabetical order: Frano Rajič, Ivan Stresec, Matea Španić, Adam Wiker
