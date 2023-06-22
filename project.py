import cv2

if __name__ == "__main__":
    # Φορτώνουμε την εικόνα.
    originalImage = cv2.imread("image.jpg")

    if originalImage is None:
        print("Failed to load the image!")
        exit()

    swapedImage = cv2.imread('image.jpg')

    # Ελέγχουμε αν η εικόνα φορτώθηκε επιτυχώς
    if swapedImage is None:
        print("Failed to load the image!")
        exit()

    # Μετατρέπουμε την εικόνα σε grey scaled.
    gray = cv2.cvtColor(swapedImage, cv2.COLOR_BGR2GRAY)

    # Φορτώνουμε το pre-trained classifier για την αναγνώριση προσώπου.
    haarCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Κάνουμε αναγνώριση προσώπου.
    faces = haarCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)

    # Προσδιορίζουμε την περιοχή του προσώπου.
    for (x, y, w, h) in faces:
        cv2.rectangle(swapedImage, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)

    # Αν δεν υπάρχουν ακριβώς 2 πρόσωπα στην εικόνα, εμφανίζει κατάλληλο μήνυμα και τερματίζει το πρόγραμμα.
    if len(faces) != 2:
        print("The number of faces in this pictures are different than 2!")
        exit()

    # Παίρνουμε τα δύο πρόσωπα.
    face1 = faces[0]
    face2 = faces[1]

    # Παίρνουμε τα 2 πρόσωπα.
    x1, y1, w1, h1 = face1
    x2, y2, w2, h2 = face2
    faceRegion1 = swapedImage[y1:y1+h1, x1:x1+w1]
    faceRegion2 = swapedImage[y2:y2+h2, x2:x2+w2]

    # Προσαρμόζουμε το μέγεθος των περιγραμμάτων των προσπώπων για να ταιριάζουν.
    faceRegion1 = cv2.resize(faceRegion1, (w2, h2))
    faceRegion2 = cv2.resize(faceRegion2, (w1, h1))

    # Αντικαθιστούμε τα πρόσωπα.
    swapedImage[y1:y1+h1, x1:x1+w1] = faceRegion2
    swapedImage[y2:y2+h2, x2:x2+w2] = faceRegion1

    # Εμφανίζουμε το αποτέλεσμα.
    cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Face Swaped', cv2.WINDOW_NORMAL)
    cv2.imshow('Face Detection', originalImage)
    cv2.imshow('Face Swaped', swapedImage)

    cv2.waitKey(0)
