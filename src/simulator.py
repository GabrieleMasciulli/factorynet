import numpy as np
import zarr
import os


def generate_industrial_data(n_samples=10000, seq_len=128, tool_is_dull=False):
    """
    Simulates motor telemetry.
    Current = (Inertia * Acceleration) + (Friction * Velocity) + Noise
    """
    # Velocity Command signal
    t = np.linspace(0, 100, n_samples * seq_len)
    v_cmd = np.sin(0.1 * t) + 0.5 * np.sin(
        0.5 * t
    )  # simulates varying industrial cycles

    # physics constants
    inertia = 1.0
    base_friction = 0.5
    dull_tool_friction = 2.5  # higher friction for dull tools
    noise_floor = 0.1

    mu = dull_tool_friction if tool_is_dull else base_friction

    # dv/dt (Acceleration)
    accel = np.gradient(v_cmd)

    # I = J*a + mu*v + noise
    current = (
        (inertia * accel) + (mu * v_cmd) + np.random.normal(0, noise_floor, v_cmd.shape)
    )

    # reshape into windows of size seq_len
    v_cmd_windows = v_cmd.reshape(-1, seq_len, 1)
    current_windows = current.reshape(-1, seq_len, 1)

    return v_cmd_windows.astype(np.float32), current_windows.astype(np.float32)


def save_to_zarr(commands, currents, filename):
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", filename)

    # Store as a structured Zarr group
    root = zarr.open_group(path, mode="w")

    root.create_array("velocity_cmd", data=commands, chunks=(100, seq_len, 1))
    root.create_array("motor_current", data=currents, chunks=(100, seq_len, 1))

    print(f"Dataset saved to {path}")


if __name__ == "__main__":
    seq_len = 128

    # generates training Data for Healthy tools
    v_train, i_train = generate_industrial_data(
        n_samples=500, seq_len=seq_len, tool_is_dull=False
    )
    save_to_zarr(v_train, i_train, "train_healthy.zarr")

    # generates test data with mixed Healthy and Dull tools
    v_test_h, i_test_h = generate_industrial_data(
        n_samples=100, seq_len=seq_len, tool_is_dull=False
    )
    v_test_d, i_test_d = generate_industrial_data(
        n_samples=100, seq_len=seq_len, tool_is_dull=True
    )

    # Stack for testing
    v_test = np.vstack([v_test_h, v_test_d])
    i_test = np.vstack([i_test_h, i_test_d])
    # Labels (0 for healthy, 1 for dull) - only for evaluation later!
    labels = np.array([0] * 100 + [1] * 100)

    save_to_zarr(v_test, i_test, "test_mixed.zarr")
    np.save("data/test_labels.npy", labels)
