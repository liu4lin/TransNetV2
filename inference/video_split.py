import subprocess
from subprocess import check_call, PIPE, Popen
import shlex
import math
import re
import os
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter

re_metadata = re.compile('Duration: (\d{2}):(\d{2}):(\d{2})\.\d+,.*\n.* (\d+(\.\d+)?) fps')
def get_metadata(filename):
    '''
    Get video metadata using ffmpeg
    '''
    p1 = Popen(["ffmpeg", "-hide_banner", "-i", filename], stderr=PIPE, universal_newlines=True)
    output = p1.communicate()[1]
    matches = re_metadata.search(output)
    if matches:
        video_length = int(matches.group(1)) * 3600 + int(matches.group(2)) * 60 + int(matches.group(3))
        video_fps = float(matches.group(4))
        # print('video_length = {}\nvideo_fps = {}'.format(video_length, video_fps))
    else:
        raise Exception("Can't parse required metadata")
    return video_length, video_fps


if __name__=='__main__':
    proj='brisbane1'
    vpath = proj+".mp4"
    spath = proj+".mp4.scenes.txt"
    odir = "clips/"+proj
    kdir1 = "kfmpg/"+proj
    kdir2 = "katna/"+proj
    if not os.path.isdir(odir):
        os.mkdir(odir)
    if not os.path.isdir(kdir1):
        os.mkdir(kdir1)
    if not os.path.isdir(kdir2):
        os.mkdir(kdir2)
    v_len, v_fps = get_metadata(vpath)
    vd = Video()
    with open(spath) as f:
        for i, line in enumerate(f.readlines()):
            start, end = [int(x) for x in line.strip().split(" ")]
            split_start = start/v_fps
            split_size = (end-start)/v_fps
            if split_size < 2: # skip the clip that is less than 2 seconds
                continue
            cpath = os.path.join(odir, str(i)+'.mp4')
            cmd = 'ffmpeg -hide_banner -loglevel panic -y -ss {} -t {} -i {} {}'.format(
                split_start, 
                split_size, 
                vpath, 
                cpath
            )
            print('Cut Video {}: {}'.format(i, cmd))
            subprocess.run(shlex.split(cmd))
            #check_call(shlex.split(cmd), universal_newlines=True)
            
            # extract key frames for a given clip via FFMpeg
            #ffmpeg -skip_frame nokey -i test.mp4 -vsync vfr -frame_pts true out-%02d.jpeg
            
            kpath = os.path.join(kdir1, str(i)+"-%02d.jpeg")
            cmd = 'ffmpeg -hide_banner -loglevel panic -skip_frame nokey -i {} -vsync vfr -frame_pts true {}'.format(
                cpath,
                kpath
            )
            print('Key frame {}: {}'.format(i, cmd))
            subprocess.run(shlex.split(cmd))
            
            
            # extract key frames for a given clip via Katna
            diskwriter = KeyFrameDiskWriter(location=kdir2)
            vd.extract_video_keyframes(no_of_frames=1, file_path=cpath, writer=diskwriter)