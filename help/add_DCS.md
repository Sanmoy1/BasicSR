## 1. Modify AICNet Architecture

First, you need to modify your AICNet to output both the super-resolved image and predicted noise:

```python
class AICNetWithNoise(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, num_feat=64, noise_ch=3):
        super(AICNetWithNoise, self).__init__()

        # Your existing AICNet architecture
        self.encoder = Encoder(num_feat=num_feat)  # 4 downsampling
        self.decoder = Decoder(num_feat=num_feat)  # 4 upsampling

        # Lightweight noise prediction head
        self.noise_head = nn.Sequential(
            nn.Conv2d(num_feat, num_feat//2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat//2, noise_ch, 3, 1, 1),
            nn.Tanh()  # Bound noise to [-1, 1]
        )

    def forward(self, x):
        features = self.encoder(x)
        sr_output = self.decoder(features)
        noise_pred = self.noise_head(features[-1])  # Use deepest features
        return sr_output, noise_pred
```

## 2. Implement Renoised Data Construction

Create the RDC logic to generate K noisy variants:

```python
class RenoisedDataConstructor:
    def __init__(self, K=8, sigma_alpha=0.1):
        self.K = K
        self.sigma_alpha = sigma_alpha

    def create_renoised_variants(self, sr_output, noise_pred, device):
        """Create K renoised variants: x_k = ŷ + α_k * n̂"""
        batch_size = sr_output.shape[0]

        # Generate α_k ~ N(0, σ_α²) with zero-mean constraint
        alphas = torch.randn(self.K, batch_size, 1, 1, 1, device=device) * self.sigma_alpha
        alphas = alphas - alphas.mean(dim=0, keepdim=True)

        variants = []
        for k in range(self.K):
            alpha_k = alphas[k]
            x_k = sr_output + alpha_k * noise_pred
            variants.append(x_k)

        return variants
```

## 3. Implement DCS Loss

Create the DCS loss that enforces consistency across variants:

```python
class DCSLoss(nn.Module):
    def __init__(self, p_init=2.0, p_final=1.5, anneal_steps=100000):
        super(DCSLoss, self).__init__()
        self.p_init = p_init
        self.p_final = p_final
        self.anneal_steps = anneal_steps
        self.current_step = 0

    def get_current_p(self):
        """Anneal p from 2.0 to 1.5 during training"""
        if self.current_step >= self.anneal_steps:
            return self.p_final

        progress = self.current_step / self.anneal_steps
        return self.p_init + progress * (self.p_final - self.p_init)

    def forward(self, variants_outputs):
        """L_DCS = (1/K) ∑ ||ŷ_k - ȳ||_p"""
        K = len(variants_outputs)
        p = self.get_current_p()

        stacked = torch.stack(variants_outputs, dim=0)
        mean_output = stacked.mean(dim=0)

        dcs_loss = 0
        for k in range(K):
            diff = torch.abs(stacked[k] - mean_output)
            if p != 1.0:
                diff = diff ** p
            dcs_loss += diff.mean()

        dcs_loss = dcs_loss / K
        self.current_step += 1

        return dcs_loss
```

## 4. Update Your Model Class

Modify your SR model to handle DCS training:

```python
class AICNetDCSModel(SRModel):
    def __init__(self, opt):
        super().__init__(opt)

        # Initialize DCS components
        self.rdc = RenoisedDataConstructor(
            K=opt['train'].get('dcs_K', 8),
            sigma_alpha=opt['train'].get('dcs_sigma_alpha', 0.1)
        )
        self.dcs_loss = DCSLoss(
            p_init=2.0, p_final=1.5,
            anneal_steps=opt['train'].get('dcs_anneal_steps', 100000)
        )

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        # Forward pass
        self.output, self.noise_pred = self.net_g(self.lq)

        l_total = 0
        loss_dict = {}

        # Standard losses for paired data
        if 'gt' in self.__dict__:
            # Your existing pixel, perceptual, FFT, SSIM, GAN losses
            pass

        # DCS loss for unpaired data
        if 'gt' not in self.__dict__ and self.noise_pred is not None:
            variants = self.rdc.create_renoised_variants(
                self.output, self.noise_pred, self.device
            )

            variants_outputs = []
            for variant in variants:
                with torch.no_grad():
                    var_out, _ = self.net_g(variant)
                    variants_outputs.append(var_out)

            l_dcs = self.dcs_loss(variants_outputs)
            l_total += opt['train'].get('dcs_weight', 1.0) * l_dcs
            loss_dict['l_dcs'] = l_dcs

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
```

## 5. Update Configuration

Add DCS parameters to your YAML config:

```yaml
# training settings
train:
  dcs_K: 8  # Number of renoised variants
  dcs_sigma_alpha: 0.1  # Gaussian noise scale
  dcs_weight: 1.0  # DCS loss weight
  dcs_anneal_steps: 100000  # Steps to anneal p-norm

  # Your existing losses...
  pixel_opt:
    type: CharbonnierLoss
    loss_weight: 1.0
  perceptual_opt:
    type: PerceptualLoss
    loss_weight: 1.0
  # ... etc
```

## 6. Key Benefits

1. **No Information Loss**: Unlike masking/downsampling, this preserves all image information
2. **Noise Robustness**: Model learns to handle input noise variations
3. **Texture Preservation**: Consistency across variants helps preserve fine details
4. **Adaptive Training**: P-norm annealing makes it robust to unknown noise patterns

## 7. Implementation Steps

1. ✅ Modify AICNet architecture with noise head
2. ✅ Implement RDC for creating renoised variants
3. ✅ Create DCS loss with p-norm annealing
4. ✅ Update model class for DCS training
5. ✅ Add configuration parameters
6. 🔄 Test with your hybrid paired+unpaired dataset

The DCS implementation follows the P2N paper's approach and should significantly improve texture preservation and noise robustness, especially for your unpaired training data. The key insight is that by enforcing consistency across multiple noisy versions of the same image, the model learns to focus on signal rather than noise.