import os
#modify directory
directory = os.getcwd()
videos = next(os.walk(directory))[1]
videos.sort()
file = open(directory + "/video_list.list", "w")
for i in videos:
    path = directory + "/"+i
    clips = next(os.walk(path))[1]
    for j in clips:
        frame_path = path + "/" + j
        file.write(frame_path)
        file.write("\n")

    

#lines = [(tmpl + ' {}').format(x['vid_name'], x['label']) for x in train + test]

file.close()
print(videos)

