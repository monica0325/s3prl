import torch
import torch.nn.functional as F

class UncertaintyQuantifier:
    def __init__(self):
        pass

    @staticmethod
    def compute_entropy(logits):
        """
        Computes entropy for a batch of logits.
        Args:
            logits (Tensor): Output logits of shape [batch_size, num_classes].
        Returns:
            Tensor: Entropy values for each sample in the batch.
        """
        probabilities = F.softmax(logits, dim=-1)  # Convert logits to probabilities
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)  # Compute entropy
        return entropy
    
    @staticmethod
    def compute_confidence(logits):
        """
        Computes confidence for a batch of logits.

        Args:
            logits (Tensor): Output logits of shape [batch_size, num_classes].

        Returns:
            Tensor: Confidence values for each sample in the batch.
        """
        probabilities = F.softmax(logits, dim=-1)  # Convert logits to probabilities
        confidence = probabilities.max(dim=-1).values  # Take the maximum probability
        return confidence
    
    @staticmethod
    def compute_variance(predictions):
        """
        Computes variance from MC Dropout predictions.
        Args:
            predictions (Tensor): Predictions from MC Dropout of shape [num_samples, batch_size, num_classes].
        Returns:
            Tensor: Variance for each sample of shape [batch_size].
        """
        variance = predictions.var(dim=0).mean(dim=-1)  # Variance over samples and classes
        return variance

    
    
    # Future methods for other uncertainty measures can be added here.

class MonteCarloDropout:
    def __init__(self, model, num_samples=10):
        """
        Initializes the MC Dropout module.

        Args:
            model (nn.Module): PyTorch model with dropout layers.
            num_samples (int): Number of stochastic forward passes.
        """
        self.model = model
        self.num_samples = num_samples

    def enable_dropout(self):
        """
        Enable dropout layers during inference.
        """
        for module in self.model.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()  # Set dropout layers to train mode

    def predict(self, inputs, lengths):
        """
        Perform multiple stochastic forward passes to get MC Dropout predictions.

        Args:
            inputs (Tensor): Input features of shape [batch_size, seq_length, feature_dim].
            lengths (Tensor): Sequence lengths for input features.

        Returns:
            Tensor: Predictions from all forward passes of shape [num_samples, batch_size, num_classes].
        """
        self.enable_dropout()
        predictions = []

        for _ in range(self.num_samples):
            with torch.no_grad():
                outputs, _ = self.model(inputs, lengths)
                predictions.append(F.softmax(outputs, dim=-1))  # Convert logits to probabilities

        return torch.stack(predictions, dim=0)  # Shape [num_samples, batch_size, num_classes]
