import matplotlib.pyplot as plt


def plot_equity_curve(equity_curve, title='Equity Curve'):
    """Plot the equity curve."""
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.show()
