import torch
import zarr
import numpy as np
import matplotlib.pyplot as plt
from model import FactoryCLIP


def load_zarr_data(filename):
    root = zarr.open(f"data/{filename}", mode="r")
    v_cmd = torch.tensor(np.array(root["velocity_cmd"]), dtype=torch.float32)
    i_cur = torch.tensor(np.array(root["motor_current"]), dtype=torch.float32)
    return v_cmd, i_cur


def evaluate():
    # load trained model
    latent_dim = 64
    model = FactoryCLIP(latent_dim=latent_dim)
    model.load_state_dict(torch.load("model_checkpoint.pt"))

    # Zero-shot detection (the evaluation)
    print("Running Zero-Shot Detection on Mixed Test Set...")
    v_test, i_test = load_zarr_data("test_mixed.zarr")
    labels = np.load("data/test_labels.npy")  # 0=Healthy, 1=Dull

    model.eval()
    with torch.no_grad():
        z_cmd_test, z_cur_test = model(v_test, i_test)

        # normalize
        z_cmd_test /= z_cmd_test.norm(dim=1, keepdim=True)
        z_cur_test /= z_cur_test.norm(dim=1, keepdim=True)

        # cosine similarities
        similarities = (z_cmd_test * z_cur_test).sum(dim=1).cpu().numpy()

    # Export Results to Zarr
    out_path = "data/results_embeddings.zarr"
    res_root = zarr.open_group(out_path, mode="w")
    res_root.create_array("cmd_latents", data=z_cmd_test.numpy())
    res_root.create_array("cur_latents", data=z_cur_test.numpy())
    res_root.create_array("similarity_scores", data=similarities)
    print(f"Latents and scores exported to {out_path}")

    # Plot histogram of similarities
    plt.figure(figsize=(10, 6))
    plt.hist(
        similarities[labels == 0],
        bins=30,
        alpha=0.5,
        label="Healthy Tool",
        color="blue",
    )
    plt.hist(
        similarities[labels == 1],
        bins=30,
        alpha=0.5,
        label="Dull Tool",
        color="red",
    )
    plt.axvline(
        x=np.percentile(similarities[labels == 0], 5),
        color="black",
        linestyle="--",
        label="95% Threshold",
    )
    plt.title("Zero-Shot Anomaly Detection via Latent Distance")
    plt.xlabel("Cosine Similarity (Command vs. Current)")
    plt.ylabel("Frequency")
    plt.legend()

    plot_path = "data/anomaly_detection_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    evaluate()
