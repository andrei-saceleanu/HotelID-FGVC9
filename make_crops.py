import os
import cv2

from tqdm import tqdm

DB_PATH = "data/train_images/"

hotel_ids = os.listdir(DB_PATH)

os.makedirs("crops_pad/", exist_ok=True)

for hotel_id in tqdm(hotel_ids):
    hotel_dir = os.path.join(DB_PATH, hotel_id)
    hotel_imgs = [os.path.join(hotel_dir, elem) for elem in os.listdir(hotel_dir)]
    os.makedirs(f"crops_pad/{hotel_id}/", exist_ok=True)
    for path in hotel_imgs:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        img_base_path = os.path.basename(path)
        if h == w:
            img = cv2.resize(img, dsize=(512, 512))
            cv2.imwrite(
                os.path.join("crops_pad", f"{hotel_id}", img_base_path), img
            )
        elif h > w:
            diff = (h - w) // 2
            cv2.imwrite(
                os.path.join("crops_pad", f"{hotel_id}", img_base_path),
                cv2.resize(
                    cv2.copyMakeBorder(img, 0, 0, diff, diff, cv2.BORDER_CONSTANT, value=0),
                    dsize=(512, 512)
                )
            )

        else:
            diff = (w - h) // 2
            cv2.imwrite(
                os.path.join("crops_pad", f"{hotel_id}", img_base_path),
                cv2.resize(
                    cv2.copyMakeBorder(img, diff, diff, 0, 0, cv2.BORDER_CONSTANT, value=0),
                    dsize=(512, 512)
                )
            )

'''
            cv2.imwrite(
                os.path.join(
                    "crops", f"{hotel_id}", img_base_path.replace(".jpg", "_0.jpg")
                ),
                cv2.resize(img[:w, :w], dsize=(512, 512))
            )
            cv2.imwrite(
                os.path.join(
                    "crops", f"{hotel_id}", img_base_path.replace(".jpg", "_1.jpg")
                ),
                cv2.resize(img[diff : diff + w, :w], dsize=(512, 512))
            )
            cv2.imwrite(
                os.path.join(
                    "crops", f"{hotel_id}", img_base_path.replace(".jpg", "_2.jpg")
                ),
                cv2.resize(img[h - diff : h - diff + w, :w], dsize=(512, 512))
            )
'''

'''
            cv2.imwrite(
                os.path.join(
                    "crops", f"{hotel_id}", img_base_path.replace(".jpg", "_0.jpg")
                ),
                cv2.resize(img[:h, :h], dsize=(512, 512))
            )
            cv2.imwrite(
                os.path.join(
                    "crops", f"{hotel_id}", img_base_path.replace(".jpg", "_1.jpg")
                ),
                cv2.resize(img[:h, diff : diff + h], dsize=(512, 512))
            )
            cv2.imwrite(
                os.path.join(
                    "crops", f"{hotel_id}", img_base_path.replace(".jpg", "_2.jpg")
                ),
                cv2.resize(img[:h, w - diff : w - diff + h], dsize=(512, 512))
            )
'''
