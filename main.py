from face_functions.FaceDetection import Face


#write path image and choose the gender
distances = Face('images/Example.JPG', 'female').main()
print(distances)
#you can check the image with mask in finalImage file