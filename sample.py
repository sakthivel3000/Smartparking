import pickle

from set_regions import SmartCarParking
from final import run,regionconversion
out_file= "S:\\college\\python\\detect\\regions.p"
video_path = "S:\\college\\python\\detect\\carvideo\\sample.mp4"
# app = SmartCarParking(video_path, out_file)
# app.run()
region_source = regionconversion(out_file)
run(video_path,region_source)
