import cv2

if __name__ == "__main__":
    # Φορτώνουμε την εικόνα.
    image = cv2.imread('image.jpg')

    haarCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = haarCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)

    if len(faces) >= 2:
        # Extract the two faces
        face1 = faces[0]
        face2 = faces[1]

        # Extract the face regions
        x1, y1, w1, h1 = face1
        x2, y2, w2, h2 = face2
        face_region1 = image[y1:y1+h1, x1:x1+w1]
        face_region2 = image[y2:y2+h2, x2:x2+w2]

        # Resize the face regions to match each other
        face_region1 = cv2.resize(face_region1, (w2, h2))
        face_region2 = cv2.resize(face_region2, (w1, h1))

        # Swap the faces
        image[y1:y1+h1, x1:x1+w1] = face_region2
        image[y2:y2+h2, x2:x2+w2] = face_region1


    cv2.imshow('Face Detection', image)

    cv2.waitKey(0)
