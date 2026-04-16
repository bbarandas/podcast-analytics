from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

EXPORTS_DIR = Path(__file__).parent.parent / "exports"


def export_png(fig: Figure, filename: str) -> Path:
    """Save a matplotlib Figure as a PNG to the exports directory.

    Args:
        fig: The matplotlib Figure to export.
        filename: Output filename (e.g. 'heatmap_messi.png').

    Returns:
        Path to the saved file.
    """
    EXPORTS_DIR.mkdir(exist_ok=True)
    output_path = EXPORTS_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
