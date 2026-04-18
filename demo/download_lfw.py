import os
os.makedirs(r'd:\collections2026\phd_application\vision_code\vision_code\demo\sample_faces\known', exist_ok=True)
os.makedirs(r'd:\collections2026\phd_application\vision_code\vision_code\demo\sample_faces\unknown', exist_ok=True)
from sklearn.datasets import fetch_lfw_people
import numpy as np
import cv2

print('Downloading LFW...')
lfw = fetch_lfw_people(min_faces_per_person=5, resize=0.5)
X = lfw.images
y = lfw.target
names = lfw.target_names

print(f'Got {len(X)} images, {len(names)} people')
print('People:', list(names))

person_img_count = {}
for idx, pid in enumerate(y):
    if pid not in person_img_count:
        person_img_count[pid] = []
    person_img_count[pid].append(idx)

known_dir = r'd:\collections2026\phd_application\vision_code\vision_code\demo\sample_faces\known'
unknown_dir = r'd:\collections2026\phd_application\vision_code\vision_code\demo\sample_faces\unknown'

known_count = 0
unknown_count = 0
for pid, indices in person_img_count.items():
    if len(indices) >= 6:
        name = names[pid].replace(' ', '_')
        for i, idx in enumerate(indices[:6]):
            img = (X[idx] * 255).astype(np.uint8)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            path = os.path.join(known_dir, f'{name}_{i}.jpg')
            cv2.imwrite(path, img)
            known_count += 1
        if len(indices) > 6:
            for i, idx in enumerate(indices[6:10]):
                img = (X[idx] * 255).astype(np.uint8)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                path = os.path.join(unknown_dir, f'{name}_test_{i}.jpg')
                cv2.imwrite(path, img)
                unknown_count += 1

print(f'Done! known: {known_count}, unknown: {unknown_count}')
