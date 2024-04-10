## Variation Auto Encoder using pytorch

In this project we will implement a variational auto encoder (VAE) using (de)convolutional neural networks ((d)CNN) using pytorch library, so to perform several image analysis tasks to the MNIST dataset of digits. We consider 50.000 training image dataset, 10.000 validation dataset and finally the last 10.000 image dataset for test purposes. The images have $28 \times 28$ dimensions and the pixel values are normalized in the range [0,1]. We transform the dataset into tensors and we pass them through a torch data-loader so to create data samples with batch-size of 256.

VAE consists of 2 parts: encoder and decoder. Both of these parts are neural networks with similar architectures. The encoder learns the distribution the parameters $\mu=\left(\mu_{1}, \ldots, \mu_{n}\right)^{T}=\mu(\mathbf{x})$ and $\sigma=\left(\sigma_{1}, \ldots, \sigma_{n}\right)^{T}=\sigma(\mathbf{x})$ in order to learn the probability distribution $q_{\phi}(Z \mid X)$ where $\mu(x)$ is the mean parameter and $\sigma(x)$ is the standard deviation of the given distribution, which can be expressed as $$q_{\phi}(Z \mid X) = \mathcal{N}\left(\mathbf{z} \mid \mu(X),\left(\sigma^{2}(X)\right)\right) = \prod_{i=1}^{D} \mathscr{N}\left(z_{i} \mid \mu_{i}, \sigma_{i}^{2}\right)$$

The decoder is used to learn the parameters $\left(m_{1}, \ldots, m_{n}\right)^{T}=\mathbf{m}(\mathbf{z})$ and $\left(s_{1}, \ldots, s_{n}\right)^{T}= \mathbf{s}(\mathbf{z})$ of the distribution $p_{\theta}(X \mid Z)$ in order to learn the probability distribution $p_{\theta}(X \mid Z)$, which can be expressed as $$p_{\theta}(X \mid Z) = \mathcal{N}\left(\mathbf{x} \mid m(Z),\left(s(Z)^{2}\right)\right) = \prod_{i=1}^{D} \mathscr{N}\left(x_{i} \mid m_{i}, s_{i}^{2}\right)$$

The loss function is expressed as an optimization (maximization) problem of the conditional log-likelihood, or the lower bound of it (ELBO) since it is very costly to compute the log-likelihood.

$$L \geq \mathbb{E}_{q(\mathbf{z} \mid \mathbf{x})} \left[ \log \frac{p(\mathbf{x} \mid \mathbf{z})}{q(\mathbf{z} \mid \mathbf{x})}\right]$$

It is proven that for maximizing the ELBO, meaning to be as close as possible to the true value, we need to minimize the KL-divergence since the difference of the $L - ELBO = K L(q(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{Z|x}))$. From that we use the equation bellow:

$$\log p(\mathbf{x}) = \mathbb{E}_{q(\mathbf{z} \mid \mathbf{x})}[\log p(\mathbf{x} \mid \mathbf{z})] + K L(q(\mathbf{Z} \mid \mathbf{x}) \| p(\mathbf{Z|x}))$$

Also, we are going to discuss and evaluate the variations of distributions that we used for our model. We tested Gaussian distribution, Beta distributions with [0, 1], Categorical distributions with discretised (binned) data,and Bernoulli distributions with re-interpreted data as probabilities for a given pixel to be black or white. In this specific task Bernoulli distribution (BD) is the most sufficient choice. Bernoulli distribution makes sense for black and white (i.e. binary) images. The Bernoulli distribution is binary, so it assumes that observations may only have two possible outcomes and this matches our dataset specification. Generalizing this statement we can see that when we use distributions with range [0, 1] we obtain better results because of the "nature" of our data. 

Finally, for the first part we will investigate the structure of the latent variables **Z**, and see how it captures structure that was implicitly present in the data. At first the latent space is two-dimensional, i.e. such that $\textbf{Z} = \mathbb{R}^2$. Then we evaluate the model in the first 1000 datapoints of our testing data, by using the encoder. We create a function that plots the $\mu(z_i)$ outputs of our encoder on a 2-dimensional plot, color-coded based on the labels of the digits $y_i$.

One variation would be to train our model on the latent space which is K-dimensional and we will reproduce the same procedure as before. In our setting we specify the dimensions to be 16. When the model is trained we use 1000 test data points and we use the encoder so to store its $\mu(z_i)$ output. Then we perform dimensionality reduction by using principality component analysis (PCA). We suppress the dataset to its 2 most important components and plot again the results with respect to the labels.

## Interpolation in the Latent Space

Rather than directly reconstruct images, one can perform operations on the encoded datapoints by working in the latent space, and then using the decoder part of the model to translate these operations to the data-domain.

In this different approach we will use linear interpolation so to obtain a sample $\mathbf{Z}$ from our decoder rather than directly reconstruct images, as we did in the previous case. Linear interpolation is a method of curve fitting using linear polynomials to construct new data points within the range $\lambda$ of a discrete set of known data points. So by predicting the values between the different points of our latent sample we aim to produce a reconstructed image. We create a function which takes as inputs to different data points of different class label and the $\lambda$ value, which represents the range of discrete points that our function will interpolate. For the given data points we learn their latent distributions from our encoder and we sample $\mathbf{z}$ and $\mathbf{z'}$. Then we can plug in those two points in the equation of linear interpolation that is given below:

$$\mathbf{z_{\lambda}} = \lambda \mathbf{z} + (1-\lambda)\mathbf{z'}$$

Finally we can use our decoder to obtain the sample $\mathbf{x_{\lambda}}$ from the learned distribution  $p(\mathbf{X|z_{\lambda}})$. This sample is by itself a reconstructed image since we have predicted/interpolated the values between the pixels of each given vector. 

We plot a figure containing a grid. On each row, the leftmost entry is $\mathbf{x}$, the rightmost entry is $\mathbf{x'}$, and the k entries in between are given by $x_{\lambda_i}$ , with $\lambda_i$ on a uniform k-partition of [0, 1]. We chose the right-hand side (RHS) as an "anchor" image and we use all the other class labels to obtain our results as shown in the notebook **latent_space_interpolation.ipynd**. We can see that for $\lambda = 0$ we interpolate exactly the point $\mathbf{x}$ and for $\lambda = 1$ we interpolate the point $\mathbf{x'}$. For all the other lambdas we can see that the interpolate images inherit features from the 2 images with different analogies. On the RHS an image looks more to the anchor image while on the LHS we can see the reconstructed images to look like their respected class-digit.

## Inference Without the Encoder

For this final step, we will attempt to estimate the distribution $p(\mathbf{Z}|\mathbf{x})$ without using the encoder, and see why this is useful. Specifically, we use the Bernoulli distribution of the previoud code to construct $p(\mathbf{X} \mid \mathbf{z})$.

We consider a single data point $\mathbf{x}$ from the testing data. A diagonal Gaussian $q(\mathbf{Z} \mid \Psi)$ with independent components over $\mathcal{Z}$ is used to estimate the distribution $p(\mathbf{Z} \mid \mathbf{x})$. $\Psi$ contains the mean vector and the diagonal of the co-variance matrix. We maximize the ELBO for $\mathbf{x}$ with respect to $\Psi$ as:

$$\Psi_{*}=\underset{\Psi}{\arg \max } \mathbb{E}_{q(\mathbf{Z} \mid \Psi)}[\log p(\mathbf{x} \mid \mathbf{z})]-\mathrm{KL}(q(\mathbf{Z} \mid \Psi) \| p(\mathbf{Z}))$$

We use the Bernoulli distribution to construct $p(\mathbf{X} \mid \mathbf{z})$, also use the interpretation of the code vae_bernoulli.ipynb. Hence, rather than the usual expression for the ELBO, our training method instead maximize:

$$-H(p(\mathbf{B} \mid \mathbf{x}), p(\mathbf{B} \mid \mathbf{z}))-\mathrm{KL}(q(\mathbf{Z} \mid \Psi) \| p(\mathbf{Z}))$$

In implementation, we set the decoder to be non-trainable and construct two trainable parameters $\mu$ and log-variance to represent the mean vector and the diagonal of the co-variance matrix of $\Psi$. We randomly initialized the value of $\mu$ and log-variance from a standard normalize distribution. We use stochastic gradient descent to optimize $\mu$ and log-variance with learning rate $=$ 0.001. To approximately evaluate $\mathbb{E}_{q(\mathbf{Z} \mid \Psi)}[\log p(\mathbf{x} \mid \mathbf{z})]$, we use one-sample Monte Carlo estimate and the reparameterization trick to estimate the value of $-H(p(\mathbf{B} \mid \mathbf{x}), p(\mathbf{B} \mid \mathbf{z}))$.

For a single data point $\mathbf{x}$, we train the parameters $\mu$ and log-variance for 500 epochs to ensure convergence. We also use the early stopping mechanism described in Task 1c with the negative ELBO value as the indicator. In this task we set \textbf{patience=50}.

We randomly choose 10 data points from the testing data, the reconstruction of them are shown in the code (we optimise Ψ separately each time). The figure in the code containing a 3-column grid where on each row, we display the original datapoint x, the reconstructed datapoint x' that we obtained, and a reconstruction x''  that we obtained as the first task (**vae_bernulli.ipynb**).