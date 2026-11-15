# Fashion-MNIST Image Parts with NMF

Using non-negative matrix factorization (NMF) to break clothing images into a small set of reusable pixel patterns, then rebuilding the images from those patterns to see how many it takes.

Originally completed as an assignment in a university unsupervised learning course.

## Overview

Each image is a 28 × 28 grayscale picture of a fashion item, stored as 784 pixel values. NMF approximates the image matrix **X** as the product of two smaller non-negative matrices, **X ≈ W · H**:

- **H** (patterns × pixels): each row is a pixel pattern, which can be displayed as a 28 × 28 image.
- **W** (images × patterns): how much each image uses each pattern.

Normalizing the rows of **W** gives a *cluster membership score* for each image, so an image can belong partly to several patterns instead of being forced into one cluster.

## Data

`fashion_mnist_sample.csv` is a random sample of **500 images** from the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. Each row has a class label and 784 pixel values (0 to 255). The 10 classes are Pullover, Sandal, Bag, Ankle boot, Coat, Shirt, T-shirt/top, Sneaker, Dress, and Trousers. The pixels are scaled to 0–1 by dividing by 255.

Fashion-MNIST is by Zalando Research and released under the MIT license: Xiao, Rasul, and Vollgraf, [*Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*](https://arxiv.org/abs/1708.07747) (2017).

## What's in the notebook

1. **Clusterability.** t-SNE projections of the images at six perplexity values (5 to 50) and two random states each, colored by item type. The plots suggest roughly 7 clusters.
2. **Choosing the number of patterns.** An elbow plot of the NMF objective for k = 1 to 45. There is no strong elbow, with at most a weak one near k = 10.
3. **NMF with k = 12.** Fits W and H, plots the t-SNE projection colored by each pattern's membership score, and shows the strongest membership score for every image.
4. **Looking at the patterns.** Each row of H displayed as a 28 × 28 image.
5. **Reconstruction.** Rebuilds the images as W · H and compares originals with reconstructions for each item type.
6. **NMF with k = 50.** Repeats the reconstruction with more patterns.

## Results

- The 28th image in the sample is about 79% pattern 8, 15% pattern 0, and 6% pattern 5, so most of its membership is concentrated in one pattern.
- Using **50 patterns reconstructs the images far better than 12**, as the side-by-side comparisons of originals and reconstructions show.
- With 50 patterns, the images with the highest weights on patterns 2 and 41 both look like boots.

## Run it

```bash
git clone https://github.com/anushreer25/fashion-mnist-nmf.git
cd fashion-mnist-nmf
pip install -r requirements.txt
jupyter notebook fashion_mnist_nmf.ipynb
```

The notebook reads `fashion_mnist_sample.csv` from the same folder. The saved outputs (plots and tables) are included, so GitHub shows them without running anything.

## Files

| File | Description |
| --- | --- |
| `fashion_mnist_nmf.ipynb` | Full analysis |
| `fashion_mnist_sample.csv` | 500-image sample of Fashion-MNIST |
| `requirements.txt` | Python dependencies |

## Tech

Python, pandas, NumPy, scikit-learn (NMF, t-SNE), seaborn, matplotlib
