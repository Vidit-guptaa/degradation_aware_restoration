import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def apply_filters(gray_img):
    gaussian = cv2.GaussianBlur(gray_img, (5, 5), 0)
    median = cv2.medianBlur(gray_img, 5)
    bilateral = cv2.bilateralFilter(gray_img, 9, 75, 75)
    return gaussian, median, bilateral

def edge_maps(gray_img):
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    canny = cv2.Canny(gray_img, 100, 200)
    return sobel, canny

def segment_otsu(gray_img):
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def morphology(binary_img):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    return opening, closing

def compute_metrics(reference, test):
    return psnr(reference, test, data_range=255), ssim(reference, test, data_range=255)

def save_image(path, image):
    cv2.imwrite(path, image)

def main():
    input_path = "input/sample.jpg"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at {input_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    noisy = add_gaussian_noise(gray, sigma=20)
    gaussian, median, bilateral = apply_filters(noisy)
    sobel, canny = edge_maps(gray)
    otsu = segment_otsu(gray)
    opening, closing = morphology(otsu)

    p_gaussian, s_gaussian = compute_metrics(gray, gaussian)
    p_median, s_median = compute_metrics(gray, median)
    p_bilateral, s_bilateral = compute_metrics(gray, bilateral)

    print("Gaussian Filter -> PSNR:", round(p_gaussian, 3), "SSIM:", round(s_gaussian, 3))
    print("Median Filter   -> PSNR:", round(p_median, 3), "SSIM:", round(s_median, 3))
    print("Bilateral Filter-> PSNR:", round(p_bilateral, 3), "SSIM:", round(s_bilateral, 3))

    save_image(os.path.join(output_dir, "original_gray.png"), gray)
    save_image(os.path.join(output_dir, "noisy.png"), noisy)
    save_image(os.path.join(output_dir, "gaussian.png"), gaussian)
    save_image(os.path.join(output_dir, "median.png"), median)
    save_image(os.path.join(output_dir, "bilateral.png"), bilateral)
    save_image(os.path.join(output_dir, "sobel.png"), sobel)
    save_image(os.path.join(output_dir, "canny.png"), canny)
    save_image(os.path.join(output_dir, "otsu.png"), otsu)
    save_image(os.path.join(output_dir, "opening.png"), opening)
    save_image(os.path.join(output_dir, "closing.png"), closing)

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))

    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title("Original Gray")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title("Noisy Image")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(gaussian, cmap='gray')
    axes[0, 2].set_title("Gaussian Filter")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(median, cmap='gray')
    axes[1, 0].set_title("Median Filter")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(bilateral, cmap='gray')
    axes[1, 1].set_title("Bilateral Filter")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(sobel, cmap='gray')
    axes[1, 2].set_title("Sobel Edge")
    axes[1, 2].axis("off")

    axes[2, 0].imshow(canny, cmap='gray')
    axes[2, 0].set_title("Canny Edge")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(otsu, cmap='gray')
    axes[2, 1].set_title("Otsu Segmentation")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(closing, cmap='gray')
    axes[2, 2].set_title("Morphology Closing")
    axes[2, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_results.png"))
    plt.show()

if __name__ == "__main__":
    main()