from .sobel_gray import detect as sobel_gray
from .sobel_channel import detect as sobel_channel

METHODS = {
    "Sobel grayscale": sobel_gray,
    "Sobel per-kanal": sobel_channel,
}