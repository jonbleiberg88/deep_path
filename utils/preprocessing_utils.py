import constants
import os
import PIL
import pandas as pd


def map_slide_to_bw(path, threshold=0.9, blur_radius=7):
    im = Image.open(path)
    # convert to grayscale
    im = im.convert('L')
    # add gaussian blur
    im = im.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    pixel_vals = np.array(im) / 255

    return (pixel_vals > threshold).astype(np.uint8)

def get_percent_whitespace(data_dir, threshold=0.9, blur_radius=7):
    whitespace = []
    for root, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                full_path = os.path.join(root, filename)
                whitespace.append((full_path, np.mean(map_slide_to_bw(full_path, threshold, blur_radius))))

    return whitespace

def threshold(whitespace, threshold = 0.9):
    for file, percent in whitespace:
        if percent > threshold:
            whitespace.remove((file, percent))
            os.remove(file)

    return whitespace

def whitespace_to_csv(whitespace, out_dir):
    df = pd.DataFrame(whitespace, columns=['file_path', 'percent_whitespace'])
    df.to_csv(f'{out_dir}/percent_whitespace.csv')


def apply_histogram_thresholding(data_dir, bw_threshold=0.9, remove_threshold=0.9,
        blur_radius=7, export=True, export_dir=None):
    if export and export_dir is None:
        print("Please specify an export directory for the output csv file")
        return

    whitespace = get_percent_whitespace(data_dir, bw_threshold, blur_radius)
    whitespace = threshold(whitespace, remove_threshold)
    if export:
        whitespace_to_csv(whitespace, export_dir)
    
