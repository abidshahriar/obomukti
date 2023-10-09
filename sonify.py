import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image

# Load an image
image_path = '/home/ryker/Desktop/to/asd.jpg'

# Convert image to a numpy array
img = np.array(Image.open(image_path).convert('L'))  # 'L' converts to grayscale

# Normalize the image data to [0, 1]
img = img.astype(np.float32) / 255.0

# Flatten the 2D image array to 1D
img_flat = img.flatten()

# Sample rate (you can adjust this according to your preference)
sr = 22050

# Convert the image to a spectrogram
spec = librosa.feature.melspectrogram(y=img_flat, sr=sr)

# Convert the spectrogram to decibels
spec_db = librosa.power_to_db(spec, ref=np.max)

# Generate an audio signal from the spectrogram
audio_signal = librosa.feature.inverse.mel_to_audio(spec)

# Plot the original image
plt.subplot(3, 1, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')

# Plot the spectrogram
plt.subplot(3, 1, 2)
plt.title('Spectrogram')
librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='mel')

# Plot the generated audio signal
plt.subplot(3, 1, 3)
plt.title('Generated Audio Signal')
librosa.display.waveshow(audio_signal, sr=sr)

plt.tight_layout()
plt.show()


# Convert the image to a spectrogram
spec = librosa.feature.melspectrogram(y=img, sr=sr)

# Convert the spectrogram to decibels
spec_db = librosa.power_to_db(spec, ref=np.max)

# Generate an audio signal from the spectrogram
audio_signal = librosa.feature.inverse.mel_to_audio(spec)

# Plot the original image
plt.subplot(3, 1, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')

# Plot the spectrogram
plt.subplot(3, 1, 2)
plt.title('Spectrogram')
librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='mel')

# Plot the generated audio signal
plt.subplot(3, 1, 3)
plt.title('Generated Audio Signal')
librosa.display.waveshow(audio_signal, sr=sr)

plt.tight_layout()
plt.show()
