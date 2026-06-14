import ffmpeg

from pathlib import Path

from backend.config import image_store_dir

# extract images from video using ffmpeg
def extract_images(video_path, fps = 1):
    try:
        print('Extracting images ....')

        video_path = Path(video_path)
        images_dir = Path(image_store_dir)/ video_path.stem

        # creating new dir for each video, if exists no error
        images_dir.mkdir(parents = True, exist_ok = True)

        # video -> images (fps = 1/x for 1 frame every x seconds, q:v: 2 for high quality images)
        ffmpeg.input(str(video_path)) \
            .filter("fps", fps=f"1/{fps}") \
            .output(str(images_dir / "image_%04d.jpg"), **{"q:v": 2}) \
            .overwrite_output() \
            .run(quiet=True) # prevents ffmpeg logs

        # collects and sorts extracted frames
        image_paths = sorted(images_dir.glob("image_*.jpg"))

        if not image_paths:
            raise RuntimeError("No images extracted from video")

        print(f"Extracted {len(image_paths)} frames: {images_dir}")

        return [str(img) for img in image_paths]

    except ffmpeg.Error as e:
        print(f"[extract_images] ffmpeg error: {e.stderr.decode()}")
        raise
    except Exception as ex:
        print('Unexpected Error while extracting images', ex)
        raise