import glob
import joblib

def create_metadata():
    result = {}
    for i, path in enumerate(sorted(glob.glob("images/*.jpg"))):
        result[i] = path
    joblib.dump(result, "metadata.job.xz", compress=3)

if __name__ == "__main__":
    create_metadata()
