# cut the first 10 seconds of a video for testing
import cv2
def cut_video(input_path, output_path, duration_sec=10):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = int(fps * duration_sec)
    current_frame = 0

    while cap.isOpened() and current_frame < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()

if __name__ == "__main__":
    cut_video("videos/seq_4camera_3_2023-06-08-12:29:08.mp4", "videos/test_clip.mp4", duration_sec=10)