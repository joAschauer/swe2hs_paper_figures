"""
Script that produces all figures except of Figure 2 (Map of stations).
"""

from pathlib import Path

import download_data
import figure_01
from calibration_and_validation_figures import (
    figure_03,
    figure_A01,
    figure_A02,
    figure_04,
    figure_05,
    figure_06,
)
import figure_07

FIGURE_DIR = Path(__file__).parent.resolve() / "created_figures"

if __name__ == "__main__":
    download_data.main()
    figure_01.main(FIGURE_DIR / "figure_01.png")
    figure_03(FIGURE_DIR / "figure_03.png")
    figure_A01(FIGURE_DIR / "figure_A01.png")
    figure_A02(FIGURE_DIR / "figure_A02.png")
    figure_04(FIGURE_DIR / "figure_04.png")
    figure_05(FIGURE_DIR / "figure_05.png")
    figure_06(FIGURE_DIR / "figure_06.png")
    figure_07.main(FIGURE_DIR / "figure_07.png")