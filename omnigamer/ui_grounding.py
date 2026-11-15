"""Screen-only button candidates; geometry is measured, labels are read by VLM."""
import numpy as np
from PIL import Image,ImageDraw


def button_candidates(frame):
    image=frame.convert("RGB").copy()
    image.thumbnail((960,960))
    a=np.asarray(image).astype(int)
    mask=(a.max(2)-a.min(2)<18)&(a.mean(2)>18)
    w,h=image.size
    tracks={};boxes=[]
    for y,row in enumerate(mask):
        row=np.convolve(row,np.ones(9),mode="same")>0
        edges=np.flatnonzero(np.diff(np.r_[False,row,False]))
        current={}
        for x,z in zip(edges[::2],edges[1::2]):
            x,z=int(x),int(z)
            if not .065*w<z-x<.85*w:continue
            key=next((k for k in tracks if abs(k[0]-x)<=4 and abs(k[1]-z)<=4),None)
            current[(x,z)]=tracks.pop(key) if key is not None else y
        for (x,z),top in tracks.items():
            if 12<=y-top<=h*.16 and (z-x)/(y-top)>2.8:
                boxes.append((x,top,z,y))
        tracks=current
    merged=[]
    for box in sorted(boxes,key=lambda b:(b[0],b[1])):
        if merged and abs(box[0]-merged[-1][0])<=4 and abs(box[2]-merged[-1][2])<=4 and 0<=box[1]-merged[-1][3]<=8:
            old=merged.pop();box=(min(old[0],box[0]),old[1],max(old[2],box[2]),box[3])
        merged.append(box)
    # Candidate rectangles are proposals, never a declaration that a UI is present.
    return [[round(x*1000/w),round(y*1000/h),round(z*1000/w),round(b*1000/h)]
            for x,y,z,b in sorted(merged,key=lambda b:(b[1],b[0]))[:40]]


def labeled_frame(frame,boxes):
    image=frame.convert("RGB").copy()
    image.thumbnail((960,960))
    draw=ImageDraw.Draw(image)
    for index,box in enumerate(boxes,1):
        x,y,z,b=[int(v*s/1000) for v,s in zip(box,(image.width,image.height,image.width,image.height))]
        draw.rectangle((x,y,z,b),outline="#ff28b4",width=2)
        draw.rectangle((x,y,x+24,y+15),fill="black")
        draw.text((x+2,y+1),str(index),fill="white")
    return image


def candidate_cards(frame,boxes):
    cards=Image.new("RGB",(960,max(1,(len(boxes)+1)//2)*90),"#181818")
    draw=ImageDraw.Draw(cards)
    for index,(x,y,z,b) in enumerate(boxes):
        # Padding includes text when segmentation found just part of a bevel.
        crop=frame.crop((int(x*frame.width/1000),max(0,int(y*frame.height/1000)-8),
                         int(z*frame.width/1000),min(frame.height,int(b*frame.height/1000)+8)))
        crop.thumbnail((430,75))
        left=index%2*480;top=index//2*90
        draw.text((left+4,top+30),str(index+1),fill="white")
        cards.paste(crop,(left+36,top+5))
    return cards


def world_selection_cues(frame):
    """Recognized world-list layout: muted Play text and actual row outline.

    This is a screen geometry adapter, not an API into the game. Unknown layouts
    return no evidence instead of guessing from actor intent or mouse hover.
    """
    boxes=button_candidates(frame)
    bottom=[b for b in boxes if 800<(b[1]+b[3])/2<980]
    play=[b for b in bottom if 220<(b[0]+b[2])/2<480 and 790<(b[1]+b[3])/2<900 and b[2]-b[0]>200]
    if len(bottom)<6 or len(play)!=1:return None,None
    x,y,z,b=play[0];w,h=frame.size
    a=np.asarray(frame.convert("RGB").crop((int(x*w/1000),int(y*h/1000),int(z*w/1000),int(b*h/1000)))).astype(int)
    median=float(np.median(a.mean(2)))
    bright=float(((a.max(2)-a.min(2)<10)&(a.mean(2)>185)).mean())
    enabled=False if median<65 and bright<.01 else (True if median>85 and bright>.025 else None)
    image=frame.convert("RGB").resize((960,round(960*h/w)))
    a=np.asarray(image).astype(int);height=a.shape[0]
    white=(a.max(2)-a.min(2)<12)&(a.mean(2)>210)
    lines=[]
    for row in range(int(height*.16),int(height*.60)):
        edges=np.flatnonzero(np.diff(np.r_[False,white[row],False]))
        lines.extend((int(l),int(r),row) for l,r in zip(edges[::2],edges[1::2]) if 240<r-l<720)
    outlined=any(abs(l-x)<4 and abs(r-z)<4 and .04*height<b-y<.18*height
                 for l,r,y in lines for x,z,b in lines)
    return outlined,enabled


def first_world_row(frame):
    """Locate a visible multi-line list entry, only in the recognized list layout."""
    if world_selection_cues(frame)==(None,None):return None
    image=frame.convert("RGB").copy();image.thumbnail((960,960))
    a=np.asarray(image).astype(int);h,w=a.shape[:2]
    white=(a.max(2)-a.min(2)<15)&(a.mean(2)>85)
    counts=white[:,int(.32*w):int(.75*w)].sum(1)
    rows=np.flatnonzero((counts>=12)&(np.arange(h)>.18*h)&(np.arange(h)<.60*h))
    groups=[]
    for y in rows:
        if groups and y-groups[-1][-1]<=9:groups[-1].append(int(y))
        else:groups.append([int(y)])
    for group in groups:
        if 18<=group[-1]-group[0]<=90:
            return [260,round((group[0]-3)*1000/h),750,round((group[-1]+3)*1000/h)]
    return None
