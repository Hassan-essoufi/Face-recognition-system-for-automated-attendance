import os
import cv2
import time
import numpy as np
from random import uniform, randint

def augment_image(image_path, person_name, output_dir, nb_imgs=150):
    """
    Generate multiple augmented images from a single person image.
    """
    output_dir = os.path.join(output_dir, person_name)
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    for i in range(nb_imgs):
        augmented = img.copy()
        
        # Random rotation (-15 to +15 degrees)
        angle = uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        augmented = cv2.warpAffine(augmented, M, (w, h))
        
        # Random horizontal flip
        if randint(0, 1):
            augmented = cv2.flip(augmented, 1)
        
        # Random brightness adjustment (0.8 to 1.2)
        factor = uniform(0.8, 1.2)
        augmented = cv2.convertScaleAbs(augmented, alpha=factor, beta=0)
        
        # Random shift
        tx = randint(-5, 5)
        ty = randint(-5, 5)
        M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
        augmented = cv2.warpAffine(augmented, M_shift, (w, h))
        
        # Save image
        out_path = os.path.join(output_dir, f"aug{i+1}.jpg")
        cv2.imwrite(out_path, augmented)


def capture_faces(person_name, output_dir="..\dataset", nb_imgs=150):
    """
    Automatic capturing of images 
    """

    person_dir = os.path.join(output_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"\nCapturing {nb_imgs} images for:{person_name}...")
    
    count = 0
    last_capture = 0
    capture_delay = 0.2
    
    while count < nb_imgs:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
        
        display = frame.copy()
        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if current_time - last_capture > capture_delay:
                face_roi = gray[y:y+h, x:x+w]
                sharpness = cv2.Laplacian(face_roi, cv2.CV_64F).var()
                
                if sharpness > 15:  
                    face = frame[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (160, 160))
                    
                    # Saving image
                    filename = f"{count:03d}.jpg"
                    cv2.imwrite(os.path.join(person_dir, filename), face_resized)
                    
                    count += 1
                    last_capture = current_time
                    
        cv2.putText(display, f"Images: {count}/{nb_imgs}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Pregression
        progress = int((count / nb_imgs) * 300)
        cv2.rectangle(display, (10, 50), (10 + progress, 60), (0, 255, 0), -1)
        
        cv2.imshow('Automatic Capture', display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n {count} images captured in: {person_dir}")

