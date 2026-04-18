import os
import sys
import cv2

known_dir = r'd:\collections2026\phd_application\vision_code\vision_code\demo\sample_faces\known'
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

person_images = {}
for f in sorted(os.listdir(known_dir)):
    if not f.endswith('.jpg'):
        continue
    name = os.path.splitext(f)[0].rsplit('_', 1)[0]
    if name not in person_images:
        person_images[name] = []
    person_images[name].append(f)

good_persons = []
for name, imgs in person_images.items():
    if len(imgs) < 6:
        continue
    detected = 0
    for img in imgs:
        path = os.path.join(known_dir, img)
        gray = cv2.imread(path, 0)
        faces = cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))
        if len(faces) > 0:
            detected += 1
    if detected >= 5:
        good_persons.append((name, imgs, detected))

print(f"高检测率人员 (>=5/6): {len(good_persons)}")
for name, imgs, detected in sorted(good_persons, key=lambda x: x[2], reverse=True)[:30]:
    print(f"  {name}: {detected}/6")
