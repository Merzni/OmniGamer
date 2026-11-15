"""Image registration hint after a turn; never a claim of game progress."""
import numpy as np
from PIL import ImageFilter


def camera_feedback(before,after):
    def sample(frame):
        w,h=frame.size
        image=frame.convert("L").crop((int(.1*w),int(.08*h),int(.9*w),int(.65*h)))
        return np.asarray(image.resize((128,64)).filter(ImageFilter.GaussianBlur(1)),dtype=float)/255
    a,b=sample(before),sample(after)
    shifts=[]
    for top,left in ((5,40),(8,25),(16,50)):
        ref=a[top:top+24,left:left+32]
        baseline=float(np.abs(ref-b[top:top+24,left:left+32]).mean())
        best=(baseline,0,0)
        for dy in range(-4,10):
            for dx in range(-24,25):
                cost=float(np.abs(ref-b[top+dy:top+24+dy,left+dx:left+32+dx]).mean())
                if cost<best[0]:best=(cost,dx,dy)
        gain=(baseline-best[0])/max(baseline,.001)
        if baseline>.004 and ref.std()>.015 and gain>.25 and abs(best[1])+abs(best[2])>=2:
            shifts.append((best[1],best[2],gain))
    consistent=[p for p in shifts if sum(abs(p[0]-q[0])<=4 and abs(p[1]-q[1])<=4 for q in shifts)>=2]
    moved=len(consistent)>=2
    dx,dy,gain=np.median(consistent,axis=0) if moved else (0,0,0)
    return {"status":"shift_observed" if moved else "unconfirmed","dx":int(dx),"dy":int(dy),
            "match_gain":round(float(gain),3),"note":"Смещение ориентиров видно; это визуальная эвристика." if moved else
            "Поворот по кадрам не подтверждён. Проверь фокус игры, ввод и чувствительность."}
