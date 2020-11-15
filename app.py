import cv2
import numpy as np
import face_recognition
import os

path = 'images'
images = []
types = os.listdir(path)
known_images = []
known_persons = []
persons_type = []

for typ in types:
    images = os.listdir(f'{path}/{typ}')
    for current_name in images:
        image = cv2.imread(f'{path}/{typ}/{current_name}')
        known_images.append(image)
        known_persons.append(os.path.splitext(current_name)[0])
        persons_type.append(typ)


def find_encodings(image_list):
    encode_list = []
    for img_arr in image_list:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        current_encode = face_recognition.face_encodings(img_arr)[0]
        encode_list.append(current_encode)
    return encode_list


if __name__ == '__main__':
    print('Encoding...')
    known_encodings = find_encodings(known_images)
    print('Encoding Complete')
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        # small_img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        small_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_locations(small_img)
        encodes = face_recognition.face_encodings(small_img, faces)
        for face_encode, face_location in zip(encodes, faces):
            matches = face_recognition.compare_faces(known_encodings, encodes[0])
            face_distance = face_recognition.face_distance(known_encodings, encodes[0])
            match_index = int(np.argmin(face_distance))
            print(face_distance)
            print(matches)
            if matches[match_index] and face_distance[match_index] < 0.4:
                name = str(known_persons[match_index]).capitalize()
                current_type = str(persons_type[match_index])
                # print(name)
                y1, x2, y2, x1 = face_location
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{current_type} {name}', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("WEB Camera", img)
        if cv2.waitKey(41) & 0xFF == ord('q'):
            break
