import os,random
from datetime import datetime
from PIL import Image,ImageDraw,ImageFont
from ultralytics import YOLO
MODEL_NAME="yolov8n.pt"
ALLOWED={".jpg",".jpeg",".png",".bmp",".gif",".webp",".tiff"}
MAX_MB=8
EMOJI={"person":"🧍‍♂️","car":"🚗","dog":"🐕","cat":"🐈","book":"📕","laptop":"💻","tv":"📺"}
def font(sz=18):
    try: return ImageFont.truetype("arial.ttf",sz)
    except: return ImageFont.load_default()
def ask_image():
    p=input("Image path:").strip().strip('"').strip("'")
    if not os.path.isfile(p): raise FileNotFoundError("Image not found.")
    if os.path.splitext(p)[1].lower() not in ALLOWED: raise ValueError("Unsupported type.")
    if os.path.getsize(p)/(1024*1024)>MAX_MB: raise ValueError("File too big (>8MB).")
    return p
def main():
    path=ask_image()
    model=YOLO(MODEL_NAME)
    results=model(path)[0]
    img=Image.open(path).convert("RGB")
    d=ImageDraw.Draw(img)
    f=font(18)
    counts={}
    for box in results.boxes:
        conf=float(box.conf[0])
        if conf<0.5: continue
        cls_id=int(box.cls[0])
        lab=model.names[cls_id]
        x1,y1,x2,y2=box.xyxy[0].tolist()
        color=tuple(random.randint(80,225) for _ in range(3))
        d.rectangle([(x1,y1),(x2,y2)],outline=color,width=3)
        txt=f"{EMOJI.get(lab.lower(),'🌟')} {lab} {conf*100:.0f}%"
        d.text((x1,y1-20),txt,font=f,fill=color)
        counts[lab]=counts.get(lab,0)+1
    out=f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    img.save(out); print(f"Saved: {out}")
    for k,v in counts.items(): print(f" · {EMOJI.get(k.lower(),'🌟')} {k}: {v}")
if __name__=="__main__": main()