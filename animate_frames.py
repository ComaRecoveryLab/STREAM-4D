import cv2
import os
import glob
import argparse

def create_video_from_images(image_folder, output_video, fps=24, label=""):
    label = f"{label}_" if label else ""
    # Get a sorted list of all the PNG images in the blender output directory
    images = sorted(glob.glob(os.path.join(image_folder, f"{label}*.png")))
    try:
        images.sort(key=lambda x: int(x.split('.')[0]))
    except:
        images.sort()

    if not images:
        print("No images found in the directory.")
        exit()

    # Read the first image to get the dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Loop through all images and add them to the video
    for image in images:
        frame = cv2.imread(image)
        video.write(frame)

    # Release the video writer
    video.release()
    print(f"Video saved to {output_video}")

def main():
    parser = argparse.ArgumentParser(description="Create a video from a sequence of images.")
    parser.add_argument("-i", "--image_folder", type=str, help="Path to the folder containing PNG images.")
    parser.add_argument("-o", "--output_video", type=str, help="Output path for the video file.")
    parser.add_argument("-f", "--fps", type=int, default=24, help="Frames per second for the output video (default: 24).")
    parser.add_argument("-l", "--label", type=str, default="", help="Session Label (optional)")
    
    args = parser.parse_args()

    create_video_from_images(args.image_folder, args.output_video, args.fps, args.label)

if __name__ == "__main__":
    main()
