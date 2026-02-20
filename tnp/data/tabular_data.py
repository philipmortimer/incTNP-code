import random
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.stats import norm
from torch import nn

from tnp.data.base import GroundTruthPredictor
from tnp.data.synthetic import SyntheticBatch, SyntheticGenerator


def pre_process(
    x: torch.Tensor,
    method="zscore",
    device=None,
    dtype=torch.float32,
    epsilon: float = 1e-8,
):
    x = x.to(device or ("cuda" if torch.cuda.is_available() else "cpu"), dtype)
    if method == "minmax":
        mn, mx = x.min(dim=1, keepdim=True).values, x.max(dim=1, keepdim=True).values
        return (x - mn) / (mx - mn + epsilon)
    elif method == "zscore":
        mean, std = (
            x.mean(dim=1, keepdim=True),
            x.std(dim=1, unbiased=False, keepdim=True),
        )
        z = (x - mean) / (std + epsilon)
        mask = std < epsilon
        z = torch.where(mask, torch.zeros_like(z), z)
        return z
    elif method == "robust":
        median = x.median(1, keepdim=True).values
        q75 = x.kthvalue(int(0.75 * x.size(dim=1)), dim=1, keepdim=True).values
        q25 = x.kthvalue(int(0.25 * x.size(dim=1)), dim=1, keepdim=True).values
        return (x - median) / (q75 - q25 + epsilon)
    elif method == "maxabs":
        return x / (x.abs().max(dim=1, keepdim=True).values + epsilon)
    elif method == "none":
        return x
    else:
        raise ValueError(f"Unknown method '{method}'")


class OutlierHandler:
    def __init__(self, device="cpu"):
        self.device = device

    def remove_outliers(
        self, x: torch.Tensor, method: str = "iqr", factor: float = 4.0
    ) -> torch.Tensor:
        if method == "iqr":
            return self._iqr_outliers(x, factor)
        elif method == "zscore":
            return self._zscore_outliers(x, factor)
        elif method == "quantile":
            return self._quantile_outliers(x, 0.01, 0.99)
        else:
            return x

    def _iqr_outliers(self, x: torch.Tensor, factor: float) -> torch.Tensor:
        q1 = torch.quantile(x, 0.25, dim=1, keepdim=True)  # Shape: (B, 1, D)
        q3 = torch.quantile(x, 0.75, dim=1, keepdim=True)  # Shape: (B, 1, D)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        x_clipped = torch.clamp(x, lower_bound, upper_bound)
        return x_clipped

    def _zscore_outliers(self, x: torch.Tensor, factor: float) -> torch.Tensor:
        mean = torch.mean(x, dim=1, keepdim=True)  # Shape: (B, 1, D)
        std = torch.std(x, dim=1, keepdim=True)  # Shape: (B, 1, D)
        std = torch.clamp(std, min=1e-8)
        lower_bound = mean - factor * std
        upper_bound = mean + factor * std
        x_clipped = torch.clamp(x, lower_bound, upper_bound)
        return x_clipped

    def _quantile_outliers(
        self, x: torch.Tensor, lower_q: float, upper_q: float
    ) -> torch.Tensor:
        lower_bound = torch.quantile(
            x, lower_q, dim=1, keepdim=True
        )  # Shape: (B, 1, D)
        upper_bound = torch.quantile(
            x, upper_q, dim=1, keepdim=True
        )  # Shape: (B, 1, D)
        x_clipped = torch.clamp(x, lower_bound, upper_bound)
        return x_clipped


def trunc_normal(mu, sigma, a, b):
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    u = np.random.uniform(norm.cdf(alpha), norm.cdf(beta))
    z = norm.ppf(u)
    return mu + sigma * z


def trunc_normal_lower(mu, sigma, a):
    alpha = (a - mu) / sigma
    u = np.random.uniform(norm.cdf(alpha), 1.0)
    z = norm.ppf(u)
    return mu + sigma * z


def convert_to_required_type(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if self.param_type == "int":
            return int(round(float(result)))
        elif self.param_type == "float":
            return float(result)
        else:
            return result

    return wrapper


class Hyperparameter:
    def __init__(self, distribution: str, param_type: Optional[str] = None, **kwargs):
        self.distribution = distribution
        self.param_type = param_type
        self.params = kwargs

    @convert_to_required_type
    def sample(self):
        if self.distribution == "uniform":
            min_uniform = self.params["min"]
            if isinstance(min_uniform, Hyperparameter):
                min_uniform = min_uniform.sample()
            max_uniform = self.params["max"]
            if isinstance(max_uniform, Hyperparameter):
                max_uniform = max_uniform.sample()
            return np.random.uniform(min_uniform, max_uniform)
        elif self.distribution == "uniform_int":
            min_uniform_int = self.params["min"]
            if isinstance(min_uniform_int, Hyperparameter):
                min_uniform_int = min_uniform_int.sample()
            max_uniform_int = self.params["max"]
            if isinstance(max_uniform_int, Hyperparameter):
                max_uniform_int = max_uniform_int.sample()
            return np.random.randint(min_uniform_int, max_uniform_int + 1)
        elif self.distribution == "log_uniform":
            min_log_uniform = self.params["min"]
            if isinstance(min_log_uniform, Hyperparameter):
                min_log_uniform = min_log_uniform.sample()
            max_log_uniform = self.params["max"]
            if isinstance(max_log_uniform, Hyperparameter):
                max_log_uniform = max_log_uniform.sample()
            return np.exp(
                np.random.uniform(np.log(min_log_uniform), np.log(max_log_uniform))
            )
        elif self.distribution == "beta":
            beta_a = self.params["a"]
            if isinstance(beta_a, Hyperparameter):
                beta_a = beta_a.sample()
            beta_b = self.params["b"]
            if isinstance(beta_b, Hyperparameter):
                beta_b = beta_b.sample()
            return np.random.beta(beta_a, beta_b)
        elif self.distribution == "scaled_beta":
            beta_a = self.params["a"]
            if isinstance(beta_a, Hyperparameter):
                beta_a = beta_a.sample()
            beta_b = self.params["b"]
            if isinstance(beta_b, Hyperparameter):
                beta_b = beta_b.sample()
            scale_term = self.params["scale_term"]
            if isinstance(scale_term, Hyperparameter):
                scale_term = scale_term.sample()
            return np.random.beta(beta_a, beta_b) * scale_term
        elif self.distribution == "gamma":
            gamma_a = self.params["a"]
            if isinstance(gamma_a, Hyperparameter):
                gamma_a = gamma_a.sample()
            gamma_scale = self.params["scale"]
            if isinstance(gamma_scale, Hyperparameter):
                gamma_scale = gamma_scale.sample()
            return np.random.gamma(gamma_a, gamma_scale)
        elif self.distribution == "normal":
            normal_mean = self.params["mean"]
            if isinstance(normal_mean, Hyperparameter):
                normal_mean = normal_mean.sample()
            normal_std = self.params["std"]
            if isinstance(normal_std, Hyperparameter):
                normal_std = normal_std.sample()
            return np.random.normal(normal_mean, normal_std)
        elif self.distribution == "lognormal":
            lognormal_mean = self.params["mean"]
            if isinstance(lognormal_mean, Hyperparameter):
                lognormal_mean = lognormal_mean.sample()
            lognormal_std = self.params["std"]
            if isinstance(lognormal_std, Hyperparameter):
                lognormal_std = lognormal_std.sample()
            return np.random.lognormal(lognormal_mean, lognormal_std)
        elif self.distribution == "truncated_normal":
            tn_mean = self.params["mean"]
            if isinstance(tn_mean, Hyperparameter):
                tn_mean = tn_mean.sample()
            tn_std = self.params["std"]
            if isinstance(tn_std, Hyperparameter):
                tn_std = tn_std.sample()
            tn_a = self.params["a"]
            if isinstance(tn_a, Hyperparameter):
                tn_a = tn_a.sample()
            if "b" not in self.params:
                return trunc_normal_lower(tn_mean, tn_std, tn_a)
            else:
                tn_b = self.params["b"]
                if isinstance(tn_b, Hyperparameter):
                    tn_b = tn_b.sample()
                return trunc_normal(tn_mean, tn_std, tn_a, tn_b)
        elif self.distribution == "choice":
            choices = self.params["choices"]
            if isinstance(choices, Hyperparameter):
                choices = choices.sample()
            return random.choice(choices)
        elif self.distribution == "zipf":
            zipf_a = self.params["a"]
            if isinstance(zipf_a, Hyperparameter):
                zipf_a = zipf_a.sample()
            zipf_max_val = self.params.get("max_val", 10)
            if isinstance(zipf_max_val, Hyperparameter):
                zipf_max_val = zipf_max_val.sample()
            return min(np.random.zipf(zipf_a), zipf_max_val)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


class LinearWithActivationAndGaussianNoise(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str,
        noise_std: float,
        device: str,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.linear_layer = nn.Linear(self.input_dim, self.output_dim, device=device)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        y = self.linear_layer(x)
        y = self.activation(y)
        return y + torch.normal(torch.zeros_like(y), self.noise_std)


class LinearWithActivation(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: str, device: str):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.linear_layer = nn.Linear(self.input_dim, self.output_dim, device=device)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        y = self.linear_layer(x)
        return self.activation(y)


class CausalMLPPrior:
    def __init__(
        self,
        num_causes: int,
        num_layers: int,
        num_features: int,
        num_outputs: int,
        hidden_dim: int,
        activation: str,
        dropout_prob: float,
        noise_std: float,
        init_std: float,
        is_causal: bool,
        categorical_feature_prob: float,
        pre_processing_method: str = "zscore",
        sampling_method: str = "mixed",
        outlier_method: str = "none",
        device: Optional[str] = None,
    ):
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.num_causes = num_causes
        self.num_layers = num_layers
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout_prob = dropout_prob
        self.noise_std = noise_std
        self.init_std = init_std
        self.is_causal = is_causal
        self.pre_processing_method = pre_processing_method
        self.layers = self._create_mlp()
        self.model = nn.Sequential(*self.layers)
        self.sampling_method = sampling_method
        self.outlier_method = outlier_method
        self.categorical_feature_prob = categorical_feature_prob
        self._init_model()

    def _create_mlp(self):
        layers: list[nn.Module] = [
            LinearWithActivation(
                self.num_causes, self.hidden_dim, "none", device=self.device
            )
        ]
        for _ in range(self.num_layers - 2):
            layers.append(
                LinearWithActivationAndGaussianNoise(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.activation,
                    self.noise_std,
                    device=self.device,
                )
            )

        if self.is_causal:
            layers.append(
                LinearWithActivationAndGaussianNoise(
                    self.hidden_dim,
                    self.hidden_dim + self.num_outputs,
                    # the additional outputs are important for the causal model
                    self.activation,
                    self.noise_std,
                    device=self.device,
                )
            )
            max_num_features = self.hidden_dim * self.num_layers
            self.num_features = min(max_num_features, self.num_features)
            self.selected_features = torch.randperm(
                max_num_features, device=self.device
            )[: self.num_features]
        else:
            layers.append(
                LinearWithActivationAndGaussianNoise(
                    self.hidden_dim,
                    self.num_outputs,
                    self.activation,
                    self.noise_std,
                    device=self.device,
                )
            )
            self.selected_features = None

        return layers

    def _init_model(self):
        layer_index = 0
        for layer in self.model:
            layer.requires_grad_(False)
            if isinstance(layer, nn.Linear):
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                nn.init.normal_(layer.weight, std=self.init_std)
                if layer_index > 0:
                    layer.weight.data *= torch.bernoulli(
                        torch.full_like(layer.weight, 1.0 - self.dropout_prob)
                    )
            layer_index += 1

    def _sample_causes(
        self, batch_size: int, seq_len: int, sampling_method: str = "normal"
    ):
        if sampling_method == "normal":
            return torch.randn(batch_size, seq_len, self.num_causes, device=self.device)
        elif sampling_method == "uniform":
            return torch.rand(batch_size, seq_len, self.num_causes, device=self.device)
        elif sampling_method == "lognormal":
            return (
                torch.distributions.LogNormal(0.0, 1.0)
                .sample((batch_size, seq_len, self.num_causes))
                .to(device=self.device)
            )
        elif sampling_method == "gamma":
            return (
                torch.distributions.Gamma(random.uniform(0.1, 10.0), 1.0)
                .sample((batch_size, seq_len, self.num_causes))
                .to(device=self.device)
            )
        elif sampling_method == "beta":
            return (
                torch.distributions.Beta(
                    random.uniform(0.1, 10.0), random.uniform(0.1, 10.0)
                )
                .sample((batch_size, seq_len, self.num_causes))
                .to(device=self.device)
            )
        elif sampling_method == "mixed":
            causes = []
            for i in range(self.num_causes):
                p = random.random()
                if p < 0.6:
                    method = random.choice(
                        ["normal", "uniform", "lognormal", "gamma", "beta"]
                    )
                    if method == "uniform":
                        cause = torch.rand(batch_size, seq_len, 1, device=self.device)
                    elif method == "lognormal":
                        cause = (
                            torch.distributions.LogNormal(0.0, 1.0)
                            .sample((batch_size, seq_len, 1))
                            .to(device=self.device)
                        )
                    elif method == "gamma":
                        cause = (
                            torch.distributions.Gamma(random.uniform(0.1, 10.0), 1.0)
                            .sample((batch_size, seq_len, 1))
                            .to(device=self.device)
                        )
                    elif method == "beta":
                        cause = (
                            torch.distributions.Beta(
                                random.uniform(0.1, 10.0), random.uniform(0.1, 10.0)
                            )
                            .sample((batch_size, seq_len, 1))
                            .to(device=self.device)
                        )
                    else:
                        cause = torch.randn(batch_size, seq_len, 1, device=self.device)
                elif p < 0.8:
                    n_classes = random.randint(2, 10)
                    probs = torch.rand(n_classes, device=self.device)
                    probs = probs / probs.sum()
                    samples = torch.multinomial(
                        probs, seq_len * batch_size, replacement=True
                    )
                    cause = samples.float().reshape(batch_size, seq_len, 1)
                else:
                    # Convert numpy zipf samples to torch tensor on correct device
                    zipf_samples = np.random.zipf(
                        2.0 + random.random() * 2, size=(batch_size, seq_len)
                    )
                    zipf_samples = np.minimum(zipf_samples, 10)
                    cause = torch.tensor(
                        zipf_samples, dtype=torch.float, device=self.device
                    ).unsqueeze(-1)
                causes.append(cause)
            return torch.cat(causes, dim=-1)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

    def _convert_to_categorical(
        self, x: torch.Tensor, categorical_features_indices: list[int]
    ) -> torch.Tensor:
        if len(categorical_features_indices) == 0:
            return x

        mask_cat = torch.zeros((x.shape[-1],), device=self.device).bool()
        mask_cat[categorical_features_indices] = True
        num_categories = [
            max(round(random.gammavariate(1, 10)), 2)
            for _ in categorical_features_indices
        ]

        x_cat = x[:, :, mask_cat]
        results = []
        for i in range(x_cat.shape[-1]):
            x_cat_i = x_cat[:, :, i : i + 1]
            category_boundaries_indices = torch.stack(
                [
                    torch.randperm(x_cat_i.shape[1], device=self.device)[
                        : num_categories[i] - 1
                    ]
                    for _ in range(x_cat_i.shape[0])
                ]
            )
            category_boundaries = x_cat_i[
                torch.arange(x.shape[0], device=self.device).unsqueeze(1),
                category_boundaries_indices,
            ].unsqueeze(1)
            categories = (x_cat_i.unsqueeze(2) > category_boundaries).sum(dim=2).float()
            results.append(categories)
        x_copy = x.clone()
        x_copy[:, :, mask_cat] = torch.cat(results, dim=-1)

        return x_copy

    def sample(self, batch_size: int, seq_len: int):
        causes = self._sample_causes(
            batch_size, seq_len, sampling_method=self.sampling_method
        )

        outputs = [causes]
        layer_index = 0
        for layer in self.layers:
            previous_output = outputs[-1]
            # previous_output = self._convert_to_categorical(previous_output, layer_index)
            # previous_output = pre_process(previous_output, self.pre_processing_method, device=self.device)
            # outputs[-1] = previous_output
            outputs.append(layer(previous_output))
            layer_index += 1

        if self.is_causal:
            outputs_flat = torch.cat(outputs[1:], -1)
            x = outputs_flat[:, :, self.selected_features]
            y = outputs_flat[:, :, -self.num_outputs :]
        else:
            x = outputs[0]
            y = outputs[-1]

        if self.categorical_feature_prob <= 1e-8:
            categorical_features_indices = []
        else:
            categorical_features_indices = []
            for i in range(x.shape[-1]):
                if random.random() < self.categorical_feature_prob:
                    categorical_features_indices.append(i)

        x = self._convert_to_categorical(x, categorical_features_indices)
        outlier_handler = OutlierHandler(self.device)
        x = outlier_handler.remove_outliers(x, method="zscore")

        return x, y


class MultiClassMultiOutputAssigner:
    def __init__(
        self,
        num_classes: Tuple[int, ...],
        assignment_type: Tuple[Optional[str], ...],
        balanced: Tuple[Optional[bool], ...],
        device: Optional[str] = None,
    ):
        if (
            len(num_classes) != len(assignment_type)
            or len(num_classes) != len(balanced)
            or len(assignment_type) != len(balanced)
        ):
            raise ValueError(
                "num_classes, assignment_type, and balanced must have same length"
            )

        self.num_classes = num_classes
        self.assignment_type = assignment_type
        self.balanced = balanced
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(len(num_classes)):
            if num_classes[i] == 2 and balanced[i] is None:
                raise ValueError(
                    "balanced cannot be None when num_classes == 2 for a certain output"
                )

    def _balanced_binarize(self, x: torch.Tensor) -> torch.Tensor:
        # return => (B, T, D)
        return (x > torch.median(x, dim=1, keepdim=True)[0]).float()

    def _data_based_boundaries_assignment(
        self, x: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        # x => (B, T, D)
        class_boundaries_indices = torch.stack(
            [
                torch.randperm(x.shape[1], device=self.device)[: num_classes - 1]
                for _ in range(x.shape[0])
            ]
        )
        # class_boundaries => (B, 1, num_classes - 1, D)
        class_boundaries = x[
            torch.arange(x.shape[0], device=self.device).unsqueeze(1),
            class_boundaries_indices,
        ].unsqueeze(1)
        # This line is equivalent to determining the interval index which x belongs to, where the interval boundaries
        # are given by class_boundaries. It just counts how many boundaries it crosses (by using the sum) to determine
        # the interval index.
        # x.unsqueeze(2) => (B, T, 1, D)
        # x.unsqueeze(2) > class_boundaries => (B, T, num_classes - 1, D)
        classes = (x.unsqueeze(2) > class_boundaries).sum(dim=2).float()
        # classes => (B, T, D)
        return classes

    def _random_boundaries_assignment(
        self, x: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        # x => (B, T, D)
        # min_values, max_values => (B, 1, D, 1)
        min_values = x.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        max_values = x.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        # class_thresholds = torch.rand(x.shape[0], num_classes - 1, device=self.device).unsqueeze(1).unsqueeze(-1)
        # class_thresholds => (B, 1, D, num_classes - 1)
        class_thresholds = torch.rand(
            x.shape[0], 1, x.shape[-1], num_classes - 1, device=self.device
        )
        class_thresholds = class_thresholds * (max_values - min_values) + min_values
        # x.unsqueeze(-1) => (B, T, D, 1)
        # x.unsqueeze(-1) > class_thresholds => (B, T, D, num_classes - 1)
        classes = (x.unsqueeze(-1) > class_thresholds).sum(dim=-1).float()
        # classes => (B, T, D)
        return classes

    def assign_classes(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.num_classes) != x.shape[-1]:
            raise ValueError("number of outputs should match the length of num_classes")

        # Ensure x is on the correct device
        x = x.to(self.device)

        result = []
        for i in range(x.shape[-1]):
            if self.num_classes[i] < 2:
                raise ValueError("Number of classes must be at least 2.")
            x_i = x[:, :, i : i + 1]
            if self.num_classes[i] == 2 and self.balanced[i]:
                result.append(self._balanced_binarize(x_i))
            elif self.assignment_type[i] == "rank":
                result.append(
                    self._data_based_boundaries_assignment(x_i, self.num_classes[i])
                )
            elif self.assignment_type[i] == "value":
                result.append(
                    self._random_boundaries_assignment(x_i, self.num_classes[i])
                )
            else:
                raise ValueError(f"Unknown assignment type: {self.assignment_type[i]}")
        return torch.cat(result, dim=-1)


class MultiClassLogitsBasedAssigner:
    def __init__(self, num_classes: int, device: Optional[str] = None):
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _logits_based_assignment(
        self, x: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        x_clamped = torch.clamp(x, min=-10, max=10)
        probs = torch.softmax(x_clamped, dim=-1)
        # The following is equivalent to sampling from the categorical distribution.
        classes = torch.multinomial(probs.reshape(-1, num_classes), 1)
        return classes.reshape(x.shape[0], x.shape[1], 1).float()

    def assign_classes(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == self.num_classes:
            raise ValueError("number of outputs should match num_classes")

        # Ensure x is on the correct device
        x = x.to(self.device)
        return self._logits_based_assignment(x, self.num_classes)


def get_default_prior_config(device):
    return {
        "num_causes": Hyperparameter("log_uniform", min=2, max=20, param_type="int"),
        "num_layers": Hyperparameter(
            "truncated_normal",
            mean=Hyperparameter("log_uniform", min=1, max=6, param_type="float"),
            std=Hyperparameter("log_uniform", min=1, max=6, param_type="float"),
            a=2,
            param_type="int",
        ),
        "num_features": Hyperparameter("uniform_int", min=2, max=20, param_type="int"),
        "num_outputs": 1,
        "hidden_dim": Hyperparameter(
            "truncated_normal",
            mean=Hyperparameter("log_uniform", min=5, max=130, param_type="float"),
            std=Hyperparameter("log_uniform", min=5, max=130, param_type="float"),
            a=4,
            param_type="int",
        ),
        "activation": Hyperparameter(
            "choice", choices=["elu", "relu", "tanh", "sigmoid", "none"]
        ),
        "dropout_prob": Hyperparameter(
            "scaled_beta",
            a=Hyperparameter("uniform", min=0.1, max=5.0, param_type="float"),
            b=Hyperparameter("uniform", min=0.1, max=5.0, param_type="float"),
            scale_term=0.9,
        ),
        "noise_std": Hyperparameter(
            "truncated_normal",
            mean=Hyperparameter("log_uniform", min=0.0001, max=0.3, param_type="float"),
            std=Hyperparameter("log_uniform", min=0.0001, max=0.3, param_type="float"),
            a=0.0,
            param_type="float",
        ),
        "init_std": Hyperparameter(
            "truncated_normal",
            mean=Hyperparameter("log_uniform", min=0.0001, max=0.3, param_type="float"),
            std=Hyperparameter("log_uniform", min=0.01, max=10.0, param_type="float"),
            a=0.0,
            param_type="float",
        ),
        "is_causal": Hyperparameter("choice", choices=[True, False]),
        "categorical_feature_prob": Hyperparameter(
            "truncated_normal",
            mean=Hyperparameter("log_uniform", min=0.0001, max=0.3, param_type="float"),
            std=Hyperparameter("log_uniform", min=0.01, max=10.0, param_type="float"),
            a=0.0,
            param_type="float",
        ),
        "pre_processing_method": Hyperparameter(
            "choice",
            choices=Hyperparameter(
                "choice", choices=[["minmax", "zscore", "robust", "maxabs"], ["none"]]
            ),
        ),
        "sampling_method": Hyperparameter(
            "choice",
            choices=["mixed", "beta", "gamma", "lognormal", "uniform", "normal"],
        ),
        "outlier_method": Hyperparameter(
            "choice",
            choices=Hyperparameter(
                "choice", choices=[["none"], ["iqr", "zscore", "quantile"]]
            ),
        ),
        "device": device,
    }


def tabular_dataset_simulator_sklearn_demo(
    config: Optional[dict] = None, device: Optional[str] = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    if config is None:
        config = {
            "causal_ml_prior": get_default_prior_config(device),
            "max_input_features": 20,
        }

    sampled_ml_prior_args = dict()

    for key in config["causal_ml_prior"]:
        param = config["causal_ml_prior"][key]
        if isinstance(param, Hyperparameter):
            sampled_ml_prior_args[key] = param.sample()
        else:
            sampled_ml_prior_args[key] = param

    prior = CausalMLPPrior(**sampled_ml_prior_args)
    x, y = prior.sample(1, 10000)

    if x.shape[-1] < config["max_input_features"]:
        zeros = torch.zeros(
            (x.shape[0], x.shape[1], config["max_input_features"] - x.shape[-1]),
            device=device,
        )
        x = torch.cat([x, zeros], axis=-1)

    balanced = random.choice([True, False])
    assignment_type = None
    if not balanced:
        assignment_type = random.choice(["rank", "value"])
    assigner = MultiClassMultiOutputAssigner(
        num_classes=(2,),
        assignment_type=(assignment_type,),
        balanced=(balanced,),
        device=device,
    )

    y = assigner.assign_classes(y)
    x = x.squeeze(0)
    y = y.squeeze(0).squeeze(-1)

    x = pre_process(x, method="zscore", device=device)

    x_numpy = x.detach().cpu().numpy()
    y_numpy = y.detach().cpu().numpy()

    x_train = x_numpy[:5000]
    x_test = x_numpy[5000:]
    y_train = y_numpy[:5000]
    y_test = y_numpy[5000:]

    from sklearn.ensemble import GradientBoostingClassifier

    clf = GradientBoostingClassifier(
        n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
    ).fit(x_train, y_train)

    print(f"Sklearn GBC score: {clf.score(x_test, y_test)}")
    print(f"Generated data shapes - X: {x.shape}, y: {y.shape}")
    print(f"Data types - X: {x.dtype}, y: {y.dtype}")
    print(f"X device: {x.device}, y device: {y.device}")

    return x, y


def create_tabular_dataset(
    batch_size: int = 1,
    seq_len: int = 10000,
    config: Optional[dict] = None,
    device: Optional[str] = None,
    problem: str = "regression",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if config is None:
        config = {
            "causal_ml_prior": get_default_prior_config(device),
            "max_input_features": 20,
        }

    sampled_ml_prior_args = dict()
    for key in config["causal_ml_prior"]:
        param = config["causal_ml_prior"][key]
        if isinstance(param, Hyperparameter):
            sampled_ml_prior_args[key] = param.sample()
        else:
            sampled_ml_prior_args[key] = param

    prior = CausalMLPPrior(**sampled_ml_prior_args)
    x, y = prior.sample(batch_size, seq_len)
    x = pre_process(x, method="zscore", device=device)

    if x.shape[-1] < config["max_input_features"]:
        zeros = torch.zeros(
            (x.shape[0], x.shape[1], config["max_input_features"] - x.shape[-1]),
            device=device,
        )
        x = torch.cat([x, zeros], axis=-1)

    if problem == "classification":
        balanced = random.choice([True, False])
        assignment_type = None
        if not balanced:
            assignment_type = random.choice(["rank", "value"])
        assigner = MultiClassMultiOutputAssigner(
            num_classes=(2,),
            assignment_type=(assignment_type,),
            balanced=(balanced,),
            device=device,
        )
        y = assigner.assign_classes(y)
        y = torch.cat([1 - y, y], axis=-1)
    else:
        outlier_handler = OutlierHandler(device)
        y = outlier_handler.remove_outliers(y, method="iqr", factor=3.0)
        y = pre_process(
            y,
            device=device,
            method=random.choice(["minmax", "zscore", "robust", "maxabs"]),
        )
    return x, y


class TabularDataGenerator(SyntheticGenerator):
    def sample_batch(
        self,
        nc: int,
        nt: int,
        batch_shape: torch.Size,
    ) -> SyntheticBatch:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x, y = create_tabular_dataset(batch_shape.numel(), nc + nt, device=device)

        xc = x[:, :nc, :]
        yc = y[:, :nc, :]
        xt = x[:, nc:, :]
        yt = y[:, nc:, :]

        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            gt_pred=None,
        )

    def __len__(self):
        return self.samples_per_epoch // self.batch_size

    def sample_inputs(
        self, nc: int, batch_shape: torch.Size, nt: Optional[int] = None
    ) -> torch.Tensor:
        return torch.empty(1, device="cpu")

    def sample_outputs(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[GroundTruthPredictor]]:
        return torch.empty(1, device="cpu")


class TabularDataGeneratorUniqueMLPPerDataset(SyntheticGenerator):
    def sample_batch(
        self,
        nc: int,
        nt: int,
        batch_shape: torch.Size,
    ) -> SyntheticBatch:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        xs, ys = [], []
        for i in range(batch_shape.numel()):
            x_, y_ = create_tabular_dataset(1, nc + nt, device=device)
            xs.append(x_)
            ys.append(y_)

        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)

        xc = x[:, :nc, :]
        yc = y[:, :nc, :]
        xt = x[:, nc:, :]
        yt = y[:, nc:, :]

        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            gt_pred=None,
        )

    def __len__(self):
        return self.samples_per_epoch // self.batch_size

    def sample_inputs(
        self, nc: int, batch_shape: torch.Size, nt: Optional[int] = None
    ) -> torch.Tensor:
        return torch.empty(1, device="cpu")

    def sample_outputs(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[GroundTruthPredictor]]:
        return torch.empty(1, device="cpu")


#
# def main():
#     # torch.manual_seed(0)
#     # random.seed(0)
#     # np.random.seed(0)
#     #
#     # for _ in range(10000):
#     #     x, y = create_tabular_dataset(10)
#     #     print(torch.max(x))
#     #     print(torch.mean(x))
#     #     print(torch.std(x))
#     #     print(torch.min(x))
#     #     print('---------------------')
#     tabular_dataset_simulator_sklearn_demo()
#
#
# if __name__ == '__main__':
#     main()
