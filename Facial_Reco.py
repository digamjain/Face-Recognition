import face_recognition
import easygui as eg
import numpy as np
import cv2
import os

#Directory containing known faces directories
knownf_dir = "known faces"

#Directory containing unknown faces
unknownf_dir = "unknown faces"

print("LOADING KNOWN FACES")

#Just some variables
known_facesf,known_namesf,answer=[],[],'n'

#Traversing through known faces directory
for name in os.listdir(knownf_dir):
    print("Folder Name -->",name)
    
    #Skip if you already has the encoding data stored in the directory for the person
    answer = input("Want to skip(y/n):")
    
    #If you choose not to skip this loop will create encoding data 
    if(answer == 'n'):
        
        #Creating a file named ed.txt to save encoding data
        handle = open(f"{knownf_dir}/{name}/ed.txt",'w')
        
        for filename in os.listdir(f"{knownf_dir}/{name}"):
            print(filename)
            
            #Try opening the image file to create encoding for if present in recognisable formats else skip
            try:
                image = face_recognition.load_image_file(f"{knownf_dir}/{name}/{filename}")
            except:
                continue
            
            #This next line creates encoding of the face at location 0
            #Increase the number of jitters for better encoding data
            encoding = face_recognition.face_encodings(image,num_jitters=1)[0]
            
            known_facesf.append(encoding)
            known_namesf.append(name)
            
            #Saving the encoding data with name of the person in ed.txt file 
            handle.write(str(encoding))
            handle.write('\n')
            handle.write(str(name))
            handle.write('\n')
        
        handle.close()
print("DONE")

#Function to retrieve the facial encoding data present in directories 
def pack():
    
    #Just emptying some variables
    known_facesf,known_namesf=[],[]
    
    #Traversing through known faces directory
    for name in os.listdir(knownf_dir):
        
        #Try to find the ed.txt file in the directory or else continue 
        try:
            
            #Open ed.txt in read mode
            handle = open(f"{knownf_dir}/{name}/ed.txt",'r') 
            
            y=np.array([])
            #Loop too split the data as required to append
            for a in handle:  
                for file in a.split():
                    file = file.lstrip('[').rstrip(']')
                    
                    '''Try to append the floating point value of the data if possible to numpy.array or else add it to the list of arrays.
                        Simply if the value in file variable is a number it is the part of the encoding and if it isn't a number, means it is the name of the person.
                        Then it is the end of the faces encoding data'''
                    try:
                        y = np.append(y,float(file))
                    except:
                        known_namesf.append(name)
                        known_facesf.append(y)
                        y=np.array([])
            handle.close()
        except:
            continue
        
        #Loop to remove empty numpy.arrays if any
        for i in known_facesf:
            if i.size == 0:
                known_facesf.pop(known_facesf.index(i))
                
    #Returns the encoding data of the various known faces and their names respectively
    return known_facesf,known_namesf
    
#Function to lower the size of the image if you have larger image files
#Not necessarily rerquired
#Copied this function from stack overflow can't find from where now so cant mention the person 

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and  grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
    
    print("Processing Unknown Faces")

#Assigning encoding data and names to variables
known_facesf,known_namesf=pack()

#Enter y if you also want to name the persons for further improvements 
#Enter n if you just want to see the results with the previous encoding data
ans = input("Want to improve results by giving names too(y/n)?\n")

for filename in os.listdir(unknownf_dir):
    print(filename)
    
    #Loading image file
    image = face_recognition.load_image_file(f"{unknownf_dir}/{filename}")
    
    #Uncommment the below funciton call only if required and choose the parameters wisely
    #image = image_resize(image,800,600)
    
    #Fucntion call to find the location of the face in the image
    #Increase or decrease the number_of_times_to_unsample value to improve accuracy
    #Test for various models like hog or cnn for better results
    locations = face_recognition.face_locations(image,number_of_times_to_upsample=2, model="hog")
    
    #Function call to get encoding for the face 
    encodings = face_recognition.face_encodings(image, locations)
    
    #Function call to convert from RGB color space to BGR 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    print(f'found {len(encodings)} face(s)')
    
    face_no=0
    write_names=[]    
    
    for face_encoding,face_location in zip(encodings,locations):
        
        #Function call to compare the face encodings and return the results
        #Change the tolerance accordingly to get the desired results
        results = face_recognition.compare_faces(known_facesf, face_encoding, tolerance = 0.45)
        
        match = None
        
        if ans=='n':
            #If a match is found in the image
            if True in results:
                
                #Find the name of the person at the particular index
                match = known_namesf[results.index(True)]
                
                print(f"Match found:{match}") 
                
                #To draw a hollow rectangle around the face of the person in the image
                ''' (face_location[3], face_location[0])_____________
                                                       |             |
                                                       |    face     |
                                                       |             |
                                                       |_____________|(face_location[1],face_location[2])'''
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1],face_location[2])
                color = (200, 200, 200)
                cv2.rectangle(image,top_left,bottom_right,color, thickness = 2)
                
                #To draw a filled rectangle for containing the name of the person
                top_left = (face_location[3]-2,face_location[2])
                bottom_right = (face_location[1]+2,face_location[2]+20)
                cv2.rectangle(image,top_left,bottom_right,color,cv2.FILLED)
                
                #To write the name at the specified co-ordinates
                cv2.putText(image,match,(face_location[3], face_location[2]+15),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),thickness = 1)
        
        if ans=='y':
            #To draw a hollow rectangle around the face of the person in the image
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1],face_location[2])
            color = (200, 200, 200)
            cv2.rectangle(image,top_left,bottom_right,color, thickness = 2)
            
            #If the face data of the person matches from any of the available face encoding data  
            if True in results:
                
                #Find the name of the person at the particular index
                match = known_namesf[results.index(True)]
                
                print(f"Match found:{match}")
                
                #To draw a filled rectangle for containing the name of the person
                top_left = (face_location[3]-2,face_location[2])
                bottom_right = (face_location[1]+2,face_location[2]+20)
                cv2.rectangle(image,top_left,bottom_right,color,cv2.FILLED)
                
                #To write the name at the specified co-ordinates 
                cv2.putText(image,match,(face_location[3], face_location[2]+15),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),thickness = 1)
            
            #If the face data of the person doesn't matches from any of the available face encoding data
            else:
                face_no+=1
                
                #To draw a filled rectangle for containing the number of the person
                top_left = (face_location[3]-2,face_location[2])
                bottom_right = (face_location[1]+2,face_location[2]+20)
                cv2.rectangle(image,top_left,bottom_right,color,cv2.FILLED)
                
                #To write the number at the specified co-ordinates 
                cv2.putText(image,str(face_no),(face_location[3], face_location[2]+15),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),thickness = 1)
                
                write_names.append(face_encoding)
    
    #Function call to show the image with all changes
    cv2.imshow(filename,image)
    
    if ans=='y':
        
        #Loop asking to enter names of the people at different numbers
        for i in range(face_no):
            
            #Function call for gui enter box
            nm = eg.enterbox("Enter name of the person at number "+str(i+1)+" or press s for skipping\n")
            
            #If name is left empty or entered s(skip)
            if nm=='s' or nm == "" :
                continue
                
            #If the name of the person is entered 
            else:
                
                path = os.path.join(knownf_dir,nm)
                
                #Loop to check if the name already exists in directories
                #If exists append in the existing ed.txt file or create a new one if ed.txt doesn't exist  
                if os.path.exists(path):
                    print("\nDirectory already exists\n")                    
                    path = os.path.join(path,'ed.txt')
                    handle = open(path,'a')
                    
                #If path doesn't create it and make a new ed.txt file for encoding data
                else:
                    os.mkdir(path)
                    print("\nNew directory created\n")
                    path = os.path.join(path,'ed.txt')
                    handle = open(path,'w')
                    
                #Saving the encoding data with name of the person in ed.txt file 
                handle.write(str(write_names[i]))
                handle.write('\n')
                handle.write(str(nm))
                handle.write('\n')
                handle.close()
        #Again calling the pack function so as new values too can get append into the variables
        known_facesf,known_namesf=pack()
    
    #Function to wait for a keyboard event
    cv2.waitKey(0)
    
    #Destroy the image window after the previous event 
    cv2.destroyWindow(filename)
print("Done")
