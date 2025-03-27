import cv2
import numpy as np
from utils import noise_reduction,line_detection,morph_operation_to_get_line,refine_boundary
### Getting Boundary ##
def boundary(room_img,saved_file):
    room_img = cv2.imread(room_img)
    noise_reduced_img = noise_reduction(room_img)
    line_Detection = line_detection(noise_reduced_img)
    morphed_img = morph_operation_to_get_line(line_Detection)
    refined_boundary = refine_boundary(morphed_img)
    cv2.imwrite(saved_file,refined_boundary)
    print("Saved file")
    return refined_boundary
if __name__ == "__main__":
    room1_path = 'room1.pgm'  # Replace with your image path
    boundary(room1_path,"Room1_wall.png")
    room2_path = 'room2.pgm'  # Replace with your image path
    boundary(room2_path,"Room2_wall.png")
    room3_path = 'room3.pgm'  # Replace with your image path
    boundary(room3_path,"Room3_wall.png")
