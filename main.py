import argparse
from remove import remove
from inpaint import inpaint

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--src", required = True, help="path to images or videos")
args = ap.parse_args()

remove(args)
inpaint(args)
