import torch
import matplotlib.pyplot as plt
from .model.memvim import MemVim


def visualize_reconstruction(inputs, reconstructed, anomaly_scores=None, idx=0):
    """
    Visualize input, reconstruction, and anomaly map for a single sample.
    """
    fig, axes = plt.subplots(1, 3 if anomaly_scores is not None else 2, figsize=(12, 4))
    inp_img = inputs[idx].detach().cpu().permute(1, 2, 0).numpy()
    rec_img = reconstructed[idx].detach().cpu().permute(1, 2, 0).numpy()
    axes[0].imshow(inp_img)
    axes[0].set_title('Input')
    axes[1].imshow(rec_img)
    axes[1].set_title('Reconstruction')
    if anomaly_scores is not None:
        # Use the first scale's anomaly map for visualization
        anomaly_map = anomaly_scores[0][idx].detach().cpu().numpy()
        axes[2].imshow(anomaly_map, cmap='hot')
        axes[2].set_title('Anomaly Score')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def test_memvim_model():
    # Create dummy input (batch of 2 RGB images, 512x512)
    batch_size = 2
    input_size = 512
    inputs = torch.randn(batch_size, 3, input_size, input_size)
    # Initialize model
    model = MemVim(
        base_model=model.MODEL_PATH,
        num_proto=512,
        temp=0.1,
        input_size=input_size,
        alpha=0.1,
        beta=0.25,
        lambda_weight=0.5,
        use_stages=[2, 3, 4],
    )
    # Forward pass
    outputs = model(inputs)
    # Compute loss
    total_loss, loss_dict = model.compute_loss(inputs, outputs)
    print('Loss breakdown:', loss_dict)
    # Visualize first sample
    visualize_reconstruction(inputs, outputs['reconstructed'], outputs['anomaly_scores'], idx=0)

if __name__ == "__main__":
    test_memvim_model()
