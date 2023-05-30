import numpy as np
import pickle
import os
import re

video_path = "/work/scitas-share/datasets/Vita/civil-459/videos/"
video_pkl_path = "pyskl/examples/videos_pkl/"
output_path = "pyskl/examples/merged/volleyball_pose_full.pkl"

videos = next(os.walk(video_pkl_path))[2]
print(videos)

#
merged = []
x_train = []
x_val = []
x_test = []

#loop through every file
for count,video in enumerate(videos):
    path = video_pkl_path +video
    print(path)
    with open(path, 'rb') as f:
        data = pickle.load(f) 
    #loop through every clip
    i_real = 0
    for i in range(len(data)):
        print(i)
        part_data = data[i]
        part_data["original_shape"] = part_data["img_shape"]
        #part_data["video"] = int(re.findall(r'\d+', video)[0])

        #look into annotation file to find the location of the players
        with open(video_path+ str(part_data["video"])+'/annotations.txt') as f:
            lines = f.readlines()
            found = False
        for j in range(len(lines)):
            line = lines[j]
            if(part_data["frame_dir"]+".jpg" == line.split(" ")[0]):
                part_data["label"]= line.split(" ")[1]
                found = True
                people = part_data["keypoint"]
                player = np.zeros(len(part_data["keypoint"]))
                #loop through every person detected
                for index in range(len(part_data["keypoint"])):
                    person = people[index]
                    #we only look at frame 20 since we only have annotations on this frame
                    keypoints = person[20]
                    x = (np.min(keypoints[:,0])+np.max(keypoints[:,0]))/2
                    y = (np.min(keypoints[:,1])+np.max(keypoints[:,1]))/2
                    bounding_boxes = re.findall(r'\b\d+\b', line)
                    #check if person is in the bounding box
                    for l in range(0,int((len(bounding_boxes)-1)/4)):
                        bounding_box_x = [int(bounding_boxes[1+4*l]),int(bounding_boxes[1+4*l])+int(bounding_boxes[3+4*l])]
                        bounding_box_y = [int(bounding_boxes[2+4*l]),int(bounding_boxes[2+4*l])+int(bounding_boxes[4+4*l])]
                        if((x>bounding_box_x[0]) and (x<bounding_box_x[1])):
                            if((y>bounding_box_y[0]) and (y<bounding_box_y[1])):
                                player[index] = 1

                print(np.sum(player),"/",len(player)) 
                #remove non players from the frame
                true_players = [true_players for true_players, x in enumerate(player) if x]          

        #array containing the index of the players
        true_players = np.array(true_players)
        true_players_original = true_players

        player_index = np.zeros((41,len(true_players_original)))
        player_index[20,:] = true_players_original

        #we start with frame 20 and compute the difference between the keypoints of each 
        # player at time t, with the keypoints of each person detected on time t-1. The goal 
        # will be to track the players, and have the index of each player be constant 
        # in time in the pkl file.
        for t in range(0,20):
            time = 20-t
            nb_people = len(part_data["keypoint"])
            difference = 1000000*np.ones((nb_people,nb_people))
            for player in true_players:
                #compute difference table
                for other_player in range(nb_people):
                    difference[int(player),other_player] = np.sum(np.abs(part_data["keypoint"][player,time] - part_data["keypoint"][other_player,time-1]))
            
            
            players_remaining = true_players
            #array that will contain the index of the players from the previous frame
            new_players = 100*np.ones(len(true_players))

            #loop through each player
            for a in range(len(true_players)):
                minimum = 100000*np.ones(len(players_remaining))
                minimum_index = np.zeros(len(players_remaining))
                #compute minimum keypoint difference between the players remaining
                for ii, b in enumerate(players_remaining):
                    m = min(difference[b,:])
                    minimum[ii] = m
                    minimum_index[ii] = difference[b,:].argmin()
                global_min = min(minimum)
                #we take the index of the players with the lower difference between 
                # their keypoints (most likely to be the same)
                first_player = players_remaining[minimum.argmin()]
                second_player = minimum_index[minimum.argmin()].astype(int)
                #difference set to very high, to be sure the same player won't be used 
                # twice for the same frame

                difference[:,second_player] = 1000000

                #if difference between keypoints is too high, sets manually the keypoints of the previous frame
                # to the keepoints of the actual frame . (in order to prevent the switch from a player to an other one)
                if(global_min >1000):
                    part_data["keypoint"][second_player,time-1] = part_data["keypoint"][first_player,time]

                players_remaining = np.delete(players_remaining,[np.where(first_player == players_remaining)[0]])
                index_new_player = np.where(first_player == true_players)[0]
                new_players[index_new_player] = int(second_player)

            true_players = new_players.astype(int)
            #register player index of previous frame
            player_index[time-1,:] = new_players.astype(int)
        
        true_players = true_players_original


        #We do the same as before, but we analyse from frame 20 to the end (frame 40)
        for time in range(20,40):
            nb_people = len(part_data["keypoint"])
            difference = 100000000*np.ones((nb_people,nb_people))
            for player in true_players:
                #compute difference table
                for other_player in range(nb_people):
                    difference[int(player),other_player] = np.sum(np.abs(part_data["keypoint"][player,time] - part_data["keypoint"][other_player,time+1]))

            players_remaining = true_players
            new_players = 100*np.ones(len(true_players))
            for a in range(len(true_players)):
                minimum = 100000*np.ones(len(players_remaining))
                minimum_index = np.zeros(len(players_remaining))
                for ii, b in enumerate(players_remaining):
                    m = min(difference[b,:])
                    minimum[ii] = m
                    minimum_index[ii] = difference[b,:].argmin()
                global_min = min(minimum)
                first_player = players_remaining[minimum.argmin()]
                second_player = minimum_index[minimum.argmin()].astype(int)
                if(global_min >1000):
                    part_data["keypoint"][second_player,time+1] = part_data["keypoint"][first_player,time]


                difference[:,second_player] = 100000000

                players_remaining = np.delete(players_remaining,[np.where(first_player == players_remaining)[0]])
                index_new_player = np.where(first_player == true_players)[0]
                new_players[index_new_player] = int(second_player)



            true_players = new_players.astype(int)
            player_index[time+1,:] = new_players

        #put everything in part_data.
        player_index = player_index.astype(int)
        new_keypoint = np.zeros((len(true_players_original),41,17,2))
        new_keypoint_score = np.zeros((len(true_players_original),41,17))
        for time in range(41):
            new_keypoint[:,time,:,:] = part_data["keypoint"][player_index[time,:],time,:,:]
            new_keypoint_score[:,time,:] = part_data["keypoint_score"][player_index[time,:],time,:]
        part_data["keypoint"] = new_keypoint
        part_data["keypoint_score"] = new_keypoint_score
        part_data["num_person_raw"] = len(part_data["keypoint"])

        #check that the annotation file was found
        if(found == False):
            print("error")
        part_data["frame_dir"] = str(part_data["video"])+"/"+ part_data["frame_dir"]
        part_data["video"] = int(part_data["video"])

        #if((part_data["num_person_raw"]<=12) and (part_data["num_person_raw"] >7)):
        if(True):
            data[i_real] = part_data
            #repartition:test/train
            test_list = np.array([4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47])
            val_list = np.array([0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51])

            if(part_data["video"] in val_list):
                x_val = x_val + [part_data["frame_dir"]]
                print("val")
            elif(part_data["video"] in test_list):
                x_test = x_test + [part_data["frame_dir"]]
                print("test")
            else:
                x_train = x_train + [part_data["frame_dir"]]
                print("train")
            i_real = i_real+1
         
    #merge
    new_data = data[:i_real]
    print(len(data),len(new_data),i_real)


    print(count)
    if(count==0):
        merged = new_data
    if(count>=1):
        merged = merged + new_data

   
split = {"xsub_train":x_train, "xsub_val": x_val, "xsub_test": x_test}

pkl_file = {"split": split, "annotations": merged}


myfile = open(output_path,"wb")
pickle.dump(pkl_file, myfile)
myfile.close()

with open(output_path, 'rb') as f:
    merged_data = pickle.load(f) 


print(len(merged_data))
print(merged_data["split"].keys())
data0 = merged_data["annotations"]
data0 = data0[0]
# print(data0)
