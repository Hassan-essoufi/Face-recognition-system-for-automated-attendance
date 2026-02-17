import os
import cv2
import numpy as np

import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1, MTCNN

def get_embedding(image_rgb):
    """
    Extract embedding from an image
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    detector = MTCNN(
    keep_all=True,  
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    device=device 
    )

    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    try:
        boxes, _ = detector.detect(image_rgb)
        
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0].astype(int)
            
            if x2 <= x1 or y2 <= y1:
                return None, None
            
            h, w = image_rgb.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            face = image_rgb[y1:y2, x1:x2]
            
            if face.shape[0] < 20 or face.shape[1] < 20:
                return None, None
            
            face_tensor = transform(face).unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = embedder(face_tensor)
            
            return embedding.cpu().numpy().flatten(), (x1, y1, x2-x1, y2-y1)
    
    except Exception as e:
        print(f"extraction Error: {e}")
    

def build_database(dataset_path="dataset"):
    """
    Create embedding's database
    """
    embeddings_db = {}
    
    if not os.path.exists(dataset_path):
        print(f" Directory not found: {dataset_path}")
        return embeddings_db
    
    persons = [p for p in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, p))]
    
    if not persons:
        print(" Directory is empty")
        return embeddings_db
    
    print(f"Treatement of {len(persons)} persons...")
    
    for person in persons:
        person_dir = os.path.join(dataset_path, person)
        embeds = []
        
        image_files = [f for f in os.listdir(person_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"  {person}: Pas d'images")
            continue
        
        print(f"  {person}: {len(image_files)} images")
        
        for img_file in image_files[:30]:  
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = get_embedding(img_rgb)
            if result is not None:
                embedding, _ = get_embedding(img_rgb)
            
            if embedding is not None:
                embeds.append(embedding)
        
        # Mean embedding
        if embeds:
            avg_embed = np.mean(embeds, axis=0)
            embeddings_db[person] = avg_embed
            print(f"{len(embeds)} valid embeddings")
        else:
            print(f"NO valid embedding")
    
    # Saving
    if embeddings_db:
        np.save("embeddings/embeddings.npy", embeddings_db)
        print(f"\n Created base : {len(embeddings_db)} persons")
    else:
        print("\n empty base!")
    
