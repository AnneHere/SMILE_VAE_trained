# SMILE: Semantic Vector Discovery with Variational Autoencoders

This project demonstrates how a Variational Autoencoder (VAE) trained on the CelebA dataset learns a latent space in which semantic facial attributes correspond to approximately linear directions. By computing the difference between mean latent representations of smiling and non-smiling faces, the project derives a smile attribute vector that enables controllable manipulation of smile intensity in unseen faces.

1. Motivation

Deep generative models often learn structured latent spaces where high-level semantic concepts emerge naturally. This project empirically verifies that a facial attribute (“smile”) can be represented as a linear direction in VAE latent space and used for interpretable image editing.

2. Objective

- Train a VAE on celebrity face images
- Encode smiling and non-smiling images into latent space
- Compute a smile attribute vector from class-wise latent means
- Apply the vector to new faces to produce “less smile” and “more smile” edits

3. Dataset

- CelebA dataset (~202,599 RGB images, 178×218)
- 40 binary facial attributes
- Smiling distribution:
- Smiling: ~96,039 images (47.5%)
- Non-smiling: ~106,560 images (52.5%)

4. Split:

- 80% training
- 20% testing

5. Preprocessing

- Decode image → tensor
- Center crop (80%)
- Resize to 128×128
- Normalize to [0,1]

6. Model Architecture

- Latent dimension: 64

Encoder
- Conv2D(64, 4×4, stride 2)
- Conv2D(128, 4×4, stride 2)
- Conv2D(256, 4×4, stride 2)
- Dense(512)
- Two heads: z_mean, z_logvar

Reparameterization: 
z = z_mean + exp(0.5 * z_logvar) * ε

Decoder
- Dense → reshape
- Conv2DTranspose(256 → 128 → 64)
- Conv2D(3, sigmoid)

Training
- Loss: Reconstruction MSE + KL Divergence
- Optimizer: Adam (lr = 1e-4)
- Epochs: 100
- Full VAE model saved for later encoding/decoding

7. Semantic Vector Computation
- Encode entire test set
- Separate latent vectors by smiling vs non-smiling label
- Randomly sample 10,000 latents from each group
- Compute means
- Compute attribute vector: v_smile = mean(smile) - mean(no_smile)
- Vector saved as: smile_vector.npy

8. Attribute Manipulation

For an unseen image with latent vector z0:

Less smile: z_minus = z0 - α * v_smile
More smile: z_plus  = z0 + α * v_smile

Decoded results show clear modulation of mouth curvature and cheek lift.

9. Results
- Consistent smile control across gender, age, and lighting
- Linear direction captures semantic meaning
- Minor artifacts (blurring) at high α values
