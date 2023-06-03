"""
Script that produces all figures except of Figure 2 (Map of stations).
"""

from pathlib import Path

import download_data
import figure_01

FIGURE_DIR = Path(__file__).parent.resolve() / "created_figures"

if __name__ == "__main__":
    download_data.main()
    figure_01.main(FIGURE_DIR / "figure_01.png")