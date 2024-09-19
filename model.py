import math

import torch
from torch import nn
from torch.nn import functional as F


import niw
import gmm
import gaussian
import dirichlet
from decoder import Decoder
from encoder import Encoder


class Model(nn.Module):
    def __init__(self, num_components, latent_dim, obs_dim, batch_dim, num_samples, encoder_layers, decoder_layers):
        super().__init__()
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_components = num_components
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.batch_dim = batch_dim
        self.obs_dim = obs_dim
        self.encoder = Encoder(encoder_layers)
        self.decoder = Decoder(decoder_layers)
        theta = _init_posterior(self.num_components, self.latent_dim)
        self.phi_mu, self.phi_cov, self.train_pi = _init_phi_gmm(theta, self.num_components)

    def forward(self, y: torch.Tensor):
        phi_enc = self.encoder.forward(y)
        x_k_samples, log_z_given_y, phi_tilde = self.e_step(phi_enc, (self.phi_mu, self.phi_cov, self.train_pi))
        y_hat = self.decoder.forward(x_k_samples)
        x_samples = _subsample_x(x_k_samples, log_z_given_y)[:, 0, :]
        return y_hat, x_k_samples, x_samples, log_z_given_y, phi_tilde

    def e_step(self, phi_enc: tuple[torch.Tensor], phi_gmm: torch.Tensor):
        eta1_phi_enc, eta2_phi_enc_diag = phi_enc
        eta2_phi_enc = torch.diag_embed(eta2_phi_enc_diag)
        eta1_phi_gmm, eta2_phi_gmm, pi_phi_gmm = _phi_gmm_to_nat_params(phi_gmm)

        log_z_given_y = _compute_log_z_given_y(eta1_phi_enc, eta2_phi_enc, eta1_phi_gmm, eta2_phi_gmm, pi_phi_gmm)

        eta2_phi_tilde = eta2_phi_enc.unsqueeze(1) + eta2_phi_gmm.unsqueeze(0)
        eta1_phi_tilde = (eta1_phi_enc.unsqueeze(1) + eta1_phi_gmm.unsqueeze(0)).unsqueeze(-1)

        x_k_samples = _sample_latents(eta1_phi_tilde, eta2_phi_tilde, self.num_samples)
        return x_k_samples, log_z_given_y, (eta1_phi_tilde, eta2_phi_tilde)

    def m_step(self, theta: tuple[torch.Tensor], x_samples: torch.Tensor, r_nk: torch.Tensor, step_size: float):
        beta_0, m_0, C_0, v_0 = niw.natural_to_standard(*theta[1:])
        alpha_0 = dirichlet.natural_to_standard(theta[0])

        alpha_k, beta_k, m_k, C_k, v_k, x_k, S_k = gmm.m_step(x_samples, r_nk, alpha_0, beta_0, m_0, C_0, v_0)

        A, b, beta, v_hat = niw.standard_to_natural(beta_k, m_k, C_k, v_k)
        alpha = dirichlet.standard_to_natural(alpha_k)
        theta_star = (alpha, A, b, beta, v_hat)

        return [
            (1 - step_size) * curr_param + step_size * param_star for (curr_param, param_star) in zip(theta, theta_star)
        ]

    def compute_elbo(self, y, y_hat, theta, phi_tilde, x_k_samps, log_z_given_phi):

        beta_k, m_k, C_k, v_k = niw.natural_to_standard(*theta[1:])
        mu, sigma = niw.expected_values((beta_k, m_k, C_k, v_k))
        eta1_theta, eta2_theta = gaussian.standard_to_natural(mu, sigma)
        alpha_k = dirichlet.natural_to_standard(theta[0])
        log_pi_given_theta = dirichlet.expected_log_pi(alpha_k)

        eta1_theta = eta1_theta.detach()
        eta2_theta = eta2_theta.detach()
        log_pi_given_theta = log_pi_given_theta.detach()

        eta1_phi_tilde, eta2_phi_tilde = phi_tilde
        N, K, L, _ = eta2_phi_tilde.shape
        eta1_phi_tilde = torch.reshape(eta1_phi_tilde, (N, K, L))

        N, K, S, L = x_k_samps.shape

        # log p(y | x, z)
        r_nk = torch.exp(log_z_given_phi)
        means_recon, var_recon = y_hat
        neg_log_prob = _gaussian_log_prob(y, means_recon, var_recon, r_nk)

        # log q(x|z, y, phi) + log q(z|y, phi)
        log_x_given_phi = gaussian.log_probability_nat_per_sample(x_k_samps, eta1_phi_tilde, eta2_phi_tilde)
        log_numerator = log_x_given_phi + log_z_given_phi.unsqueeze(2)

        # log p(x| z, theta) + log p(z|theta)
        log_x_given_theta = gaussian.log_probability_nat_per_sample(
            x_k_samps, eta1_theta.unsqueeze(0).expand(N, -1, -1), eta2_theta.expand(N, -1, -1, -1)
        )
        log_denominator = log_x_given_theta + log_pi_given_theta.unsqueeze(0).unsqueeze(2)

        kl_div = r_nk.unsqueeze(2) * (log_numerator - log_denominator)
        kl_div = torch.sum(kl_div, dim=1)
        kl_div = torch.sum(kl_div, dim=0)
        kl_div = torch.mean(kl_div)

        return -1.0 * (neg_log_prob - kl_div)

    def init_posterior(self, num_components: int, latent_dim: int):
        return _init_posterior(num_components, latent_dim)

    @property
    def phi_gmm(self):
        return (self.phi_mu, self.phi_cov, self.train_pi)


def _compute_log_z_given_y(eta1_phi1, eta2_phi1, eta1_phi2, eta2_phi2, pi_phi2):
    B, L = eta1_phi1.shape
    K, _ = eta1_phi2.shape

    eta2_phi_tilde = eta2_phi1.unsqueeze(1) + eta2_phi2.unsqueeze(0)
    inv_eta2_eta2_sum_eta1 = eta2_phi_tilde.inverse() @ eta2_phi2.expand(B, -1, -1, -1)
    w_eta2 = torch.einsum("nju,nkui->nkij", eta2_phi1, inv_eta2_eta2_sum_eta1)

    w_eta2 = (w_eta2 + w_eta2.transpose(dim0=-1, dim1=-2)) / 2.0
    mu_eta2_1_eta2_2 = eta2_phi_tilde.inverse() @ eta1_phi2.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 1)
    w_eta1 = torch.einsum("nuj,nkuv->nkj", eta2_phi1, mu_eta2_1_eta2_2)  # Shape: NxKxL

    mu_phi1, _ = gaussian.natural_to_standard(eta1_phi1, eta2_phi1)
    return gaussian.log_probability_nat(mu_phi1, w_eta1, w_eta2, pi_phi2)


def _gaussian_log_prob(y, param1_recon, param2_recon, weights):
    M, K, S, L = param1_recon.shape

    y = y.unsqueeze(1).unsqueeze(1)
    sample_mean = torch.einsum(
        "nksd,nk->", torch.pow(y - param1_recon, 2) / param2_recon + torch.log(param2_recon + 1e-8), weights
    )
    return -0.5 * (sample_mean / S) - M * L / 2.0 * math.log(2.0 * math.pi)


def _init_posterior(num_components: int, latent_dim: int):
    alpha_scale = 1.0
    beta_scale = 1.0
    m_scale = 5.0
    v_init = latent_dim + 1.0
    c_scale = 2 * latent_dim

    alpha_init = alpha_scale * torch.ones(num_components)
    beta_init = beta_scale * torch.ones(num_components)
    v_init = torch.tensor([float(latent_dim + v_init)]).expand(num_components)
    means_init = m_scale * torch.empty(num_components, latent_dim).uniform_(-1, 1)
    covariance_init = c_scale * torch.eye(latent_dim).expand(num_components, -1, -1)

    A, b, beta, v_hat = niw.standard_to_natural(beta_init, means_init, covariance_init, v_init)
    alpha = dirichlet.standard_to_natural(alpha_init)

    return alpha, A, b, beta, v_hat


def _init_phi_gmm(theta, num_components):
    theta = niw.natural_to_standard(theta[1].clone(), theta[2].clone(), theta[3].clone(), theta[4].clone())

    mu_k, sigma_k = niw.expected_values(theta)
    L_k = torch.linalg.cholesky(sigma_k)

    pi_k = torch.randn((num_components,))
    return nn.Parameter(mu_k), nn.Parameter(L_k), nn.Parameter(F.log_softmax(pi_k, dim=0))


def _phi_gmm_to_nat_params(phi_gmm):
    eta1, L_k_raw, pi_k_raw = phi_gmm

    L_k = torch.tril(L_k_raw)
    diag_L_k = torch.diagonal(L_k, dim1=-2, dim2=-1)
    softplus_L_k = F.softplus(diag_L_k)

    mask = torch.diag_embed(torch.ones_like(softplus_L_k))
    L_k = torch.diag_embed(softplus_L_k) + (1.0 - mask) * L_k
    precision = L_k @ torch.transpose(L_k, 2, 1)

    eta2 = -0.5 * precision
    pi_k = F.log_softmax(pi_k_raw, dim=0)

    return (eta1, eta2, pi_k)


def _sample_latents(eta1: torch.Tensor, eta2: torch.Tensor, num_samples: int):
    B, K, L, _ = eta1.shape

    inv_sigma = -2.0 * eta2
    chol_inv_sigma = torch.linalg.cholesky(inv_sigma)
    noise = chol_inv_sigma.transpose(dim0=3, dim1=2).inverse() @ torch.randn(B, K, L, num_samples)

    x_samples = inv_sigma.inverse() @ eta1 + noise
    return x_samples.permute(0, 1, 3, 2)


def _subsample_x(x_k_samples, log_q_z_given_y):
    B, K, S, L = x_k_samples.shape

    n_idx = torch.arange(start=0, end=B).unsqueeze(1).repeat(1, S)
    s_idx = torch.arange(start=0, end=S).unsqueeze(0).repeat(B, 1)

    m = torch.distributions.Categorical(logits=log_q_z_given_y)
    z_samples = torch.transpose(m.sample([S]), dim0=1, dim1=0)

    return x_k_samples[[n_idx, z_samples, s_idx]]