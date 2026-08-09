"""Compatibility entry point for the centre-only locator training stage."""

import sys

from train_pipeline import main


if __name__ == "__main__":
    main(["fit-locator", *sys.argv[1:]])
