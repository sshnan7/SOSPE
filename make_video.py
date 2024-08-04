import os
import numpy
import cv2

def make_video() :

    path = '/home/sshnan7/work_space/nlos/self_supervised_learning/results/vis/35/'
    
    file_num_list = list(range(1700, 1740))
    frame_list = []
    print(file_num_list)
    for file_num in file_num_list :
        people_num = 0
        file_path = path + '_{}_pose_num{}.png'.format(file_num, people_num)
        while os.path.isfile(file_path) == False :
            people_num += 1
            file_path = path + '_{}_pose_num{}.png'.format(file_num, people_num)
        file_img = cv2.imread(file_path)
        hei, wid = file_img.shape[0], file_img.shape[1]
        frame_list.append(file_img)
    print(file_num_list)
  
    
    size = (wid, hei)
    print(size)
   

    result = cv2.VideoWriter('two_people.avi', 
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             2, size)
    for i in frame_list :
        result.write(i)
  
    # When everything done, release 
    # the video capture and video 
    # write objects
    result.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()
    
make_video()
   