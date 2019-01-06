from nkpop import nkpop_list
from pytube import YouTube
import ffmpeg
import os, glob, tarfile
from tqdm import tqdm

def download_all():
    if not os.path.exists("videos"):
        os.mkdir("videos")
    for path in tqdm(nkpop_list):
        try:
            yt = YouTube(path)
            yt.streams.filter(fps=30, res="1080p", subtype="mp4").first().download("videos")
        except:
            continue

def convert_to_image():
    cnt = 0
    for path in tqdm(sorted(glob.glob("videos/*"))):
        if not os.path.exists("images"):
            os.mkdir("images")
        stream = ffmpeg.input(path)
        stream = ffmpeg.output(stream, f"images/{cnt:02d}_%05d.jpg",r=1 ,f="image2", q=4)
        ffmpeg.run(stream)
        cnt += 1

def pack_tar():
    with tarfile.open("images.tar", "w") as tar:
        tar.add("images")

if __name__ == "__main__":
    download_all()
    #convert_to_image()
    #pack_tar()
    
