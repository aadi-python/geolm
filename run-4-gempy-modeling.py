import numpy as np
from hutton_lm.cli import main

if __name__ == "__main__":
    # Set seed for reproducibility here as well,
    # in case the package is run via this script.
    np.random.seed(1515)
    main()
