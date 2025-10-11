"""
Demonstration of PARSIM algorithms in the new SIPPY architecture.

This example shows how to use the three PARSIM algorithms (PARSIM-K, PARSIM-S, PARSIM-P)
with the modern object-oriented identification interface.
"""
import matplotlib.pyplot as plt
import numpy as np

from sippy.identification import SystemIdentification, SystemIdentificationConfig


def generate_sample_data(N=500):
    """Generate sample data for system identification.

    Creates a second-order system with known dynamics.
    """
    np.random.seed(42)

    # Time vector
    t = np.arange(N)

    # Generate input signal (step + noise)
    u = np.ones((2, N))
    u[0, :] += 0.1 * np.random.randn(N)  # Add some noise
    u[1, :] = np.sin(0.1 * t) + 0.1 * np.random.randn(N)

    # True system matrices (second order)
    A_true = np.array([[0.8, -0.4], [0.4, 0.8]])
    B_true = np.array([[0.5, 0.2], [0.1, 0.6]])
    C_true = np.array([[1.0, 0.0]])
    D_true = np.array([[0.0, 0.0]])

    # Simulate the system
    x = np.zeros((2, N + 1))
    y = np.zeros((1, N))

    for k in range(N):
        # State update
        x[:, k + 1] = A_true @ x[:, k] + B_true @ u[:, k]
        # Output
        y[:, k] = C_true @ x[:, k] + D_true @ u[:, k] + 0.05 * np.random.randn()

    return y, u, (A_true, B_true, C_true, D_true)


def test_parsim_algorithms():
    """Test all three PARSIM algorithms."""

    # Generate sample data
    print("Generating sample data...")
    y, u, true_system = generate_sample_data(500)
    A_true, B_true, C_true, D_true = true_system

    print(f"Data shapes: y={y.shape}, u={u.shape}")

    # Configuration for PARSIM algorithms
    config = SystemIdentificationConfig(
        ss_threshold=0.05,
        ss_fixed_order=2,  # We know it's a 2nd order system
        ss_d_required=True
    )

    # List of algorithms to test
    algorithms = ['PARSIM-K', 'PARSIM-S', 'PARSIM-P']
    results = {}

    print("\nTesting PARSIM algorithms...")
    print("=" * 50)

    for algo_name in algorithms:
        print(f"\nRunning {algo_name} algorithm...")

        try:
            # Create configuration with specific algorithm
            algo_config = SystemIdentificationConfig(
                method=algo_name,
                ss_threshold=config.ss_threshold,
                ss_fixed_order=config.ss_fixed_order,
                ss_d_required=config.ss_d_required
            )

            # Perform identification
            identifier = SystemIdentification(algo_config)
            model = identifier.identify(y=y, u=u)

            # Store results
            results[algo_name] = model

            # Display results
            print(f"  Identified system order: {model.n}")
            print(f"  Noise variance: {model.Vn:.6f}")
            print(f"  System matrix A shape: {model.A.shape}")
            print(f"  A matrix eigenvalues: {np.linalg.eigvals(model.A)}")
            print(f"  System stable: {'Yes' if model.is_stable() else 'No'}")

        except Exception as e:
            print(f"  Failed with error: {str(e)}")
            results[algo_name] = None

    return results, y, u, true_system


def compare_algorithms(results, y, u, true_system):
    """Compare the performance of different algorithms."""

    print("\nAlgorithm Comparison")
    print("=" * 50)

    A_true, B_true, C_true, D_true = true_system

    for algo_name, model in results.items():
        if model is None:
            print(f"{algo_name:12s}: FAILED")
            continue

        # Compare system order
        print(f"{algo_name:12s}: Order={model.n:2d}, Vn={model.Vn:.4f}, Stable={'Y' if model.is_stable() else 'N'}")

    # Plot comparisons if matplotlib is available
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('PARSIM Algorithm Comparison', fontsize=16)

        # Plot input signals
        axes[0, 0].plot(u[0, :], label='Input 1')
        axes[0, 0].plot(u[1, :], label='Input 2')
        axes[0, 0].set_title('Input Signals')
        axes[0, 0].set_xlabel('Time step')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot output signal
        axes[0, 1].plot(y[0, :], 'b-', label='Actual Output', linewidth=2)
        axes[0, 1].set_title('Output Signal')
        axes[0, 1].set_xlabel('Time step')
        axes[0, 1].set_ylabel('Output')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Compare models if we have successful results
        successful_algos = [(name, model) for name, model in results.items() if model is not None]

        if successful_algos:
            # Plot noise variance comparison
            algo_names = [name for name, _ in successful_algos]
            vn_values = [model.Vn for _, model in successful_algos]

            bars = axes[1, 0].bar(algo_names, vn_values, color=['red', 'blue', 'green'][:len(algo_names)])
            axes[1, 0].set_title('Noise Variance Comparison')
            axes[1, 0].set_ylabel('Vn')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, vn in zip(bars, vn_values):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{vn:.3f}', ha='center', va='bottom')

            # System order comparison
            orders = [model.n for _, model in successful_algos]
            axes[1, 1].bar(algo_names, orders, color=['orange', 'purple', 'cyan'][:len(algo_names)])
            axes[1, 1].set_title('Identified Order')
            axes[1, 1].set_ylabel('System Order')
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\nMatplotlib not available for plotting")


def main():
    """Main function to run the PARSIM demonstration."""
    print("SIPPY PARSIM Algorithms Demonstration")
    print("=" * 50)

    try:
        # Run the tests
        results, y, u, true_system = test_parsim_algorithms()

        # Compare results
        compare_algorithms(results, y, u, true_system)

        print("\n" + "=" * 50)
        print("Demonstration completed successfully!")

        # Show available algorithms
        from sippy.identification.factory import AlgorithmFactory
        available_algos = AlgorithmFactory.list_algorithms()
        print(f"\nAvailable algorithms: {available_algos}")

    except Exception as e:
        print(f"\nDemonstration failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
