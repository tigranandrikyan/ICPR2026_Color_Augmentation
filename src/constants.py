import colorsys

# Hyperparameters that can be tuned depending on image size and content


# -----------------------
# Growing Neural Gas (GNG)
# -----------------------

# Number of training epochs for GNG
EPOCH = 10

# Threshold for removing edges between groups
EDGE_CUTTING = 0.65

# Number of initial nodes
STARTING_NODES = 3

# Maximum number of nodes allowed
MAX_NODES = 20

# Enable or disable Gaussian smoothing
USE_SMOOTH = False


# -----------------------
# Data settings
# -----------------------

# Maximum RGB color value
MAX_COLOR_VALUE = 255.0

# Input image file type (case-sensitive on Unix-based systems)
FILE_TYPE = ".jpg"

# Number of augmented images to generate
AUG_COUNT = 5


# -----------------------
# Fancy PCA parameters
# -----------------------

# Mean value for alpha sampling
FANCY_PCA_MEAN = 0

# Standard deviation for alpha sampling
# Higher values result in stronger color perturbations
FANCY_PCA_STANDARD_DEVIATION = 5


# -----------------------
# Gaussian smoothing (not used)
# -----------------------

# Standard deviation for Gaussian filter
SIGMA = 3


# -----------------------
# Color jitter parameters
# -----------------------

BRIGHTNESS = 0.4
CONTRAST = 0.4
SATURATION = 0.4
HUE = 0.1


# Generate a distinct RGB color for a given group index
# Colors are distributed evenly using the golden ratio
def get_color(group):
    golden_ratio_conjugate = 0.618033988749895

    # Compute hue value based on group index
    hue = (group * golden_ratio_conjugate) % 1.0

    # Convert HSV color to RGB
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 1.0)

    # Scale RGB values to [0, 255]
    return int(r * 255), int(g * 255), int(b * 255)
