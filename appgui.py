from tkinter import *
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import shutil

root = Tk()


canvas = Canvas(root, height= 850, width= 750)
canvas.pack()

def transform_files():
  label4["text"] = "running"
  filtr = entry.get()
  img_path = entry2.get()
  dst_path = entry3.get()
  keys = {0: "buildings", 1: "forest", 2: "glacier", 3: "mountains", 4: "sea", 5: "street"}
  m = load_model("Scenes.h5")
  
  try:
      os.mkdir(dst_path)

  except:
    pass
    
  for x in os.listdir(img_path):
    full_path = os.path.join(img_path, x)
    try:
        img = image.load_img(full_path, target_size=(224, 224))
        q = image.img_to_array(img)
        q = np.expand_dims(q, axis= 0)
        l = m.predict(q)
        b = np.argmax(l[0])
        f = keys[b]
        if f == filtr:
          shutil.copy(full_path, dst_path)
      
        else:
          pass
        
    except:
          pass
  
  
  label4["text"] = "Succesfully done"

def clear():
  entry.delete(0, END)
  entry2.delete(0, END)
  entry3.delete(0, END)




bimage = PhotoImage(file="D:\\okelol.png")
bc = Label(root, image=bimage)
bc.place(relwidth=1, relheight=1)

frame = Frame(root, bg = "#02bcff",bd = 4)
frame.place(relx = 0.5, rely = 0.1, relwidth = 0.65, relheight = 0.1, anchor ="n")


label = Label(frame, text="Enter picture filter: ")
label.place(relwidth= 0.37, relheight = 1)

entry = Entry(frame, font = 30)
entry.place(relx = 0.39, relwidth = 0.61, relheight = 1)

frame2 = Frame(root, bg = "#02bcff", bd=4)
frame2.place(relx = 0.5, rely = 0.25, relwidth = 0.65, relheight = 0.1, anchor ="n")

label2 = Label(frame2, text="Enter directory of images: ")
label2.place(relwidth= 0.4, relheight = 1)

entry2 = Entry(frame2, font = 30)
entry2.place(relx = 0.42, relwidth = 0.58, relheight = 1)

frame3 = Frame(root, bg = "#02bcff", bd=4)
frame3.place(relx = 0.5, rely = 0.4, relwidth = 0.65, relheight = 0.1, anchor ="n")

label3 = Label(frame3, text="Enter destination directory: ")
label3.place(relwidth= 0.48, relheight = 1)


entry3 = Entry(frame3, font = 30)
entry3.place(relx = 0.50, relwidth = 0.50, relheight = 1)

frame4 = Frame(root, bg = "#100500", bd=6)
frame4.place(relx = 0.2, rely = 0.65, relwidth = 0.35, relheight = 0.6, anchor ="n")

label4 =Label(frame4)
label4.place(relwidth = 1, relheight = 1)

frame5 = Frame(root, bg = "#100500", bd=6)
frame5.place(relx = 0.55, rely = 0.8, relwidth = 0.2, relheight = 0.15, anchor ="n")

button = Button(frame5, text = "Start", font = 40, command = transform_files)
button.place(relwidth = 1, relheight = 1)

frame6 = Frame(root, bg = "#100500", bd=6)
frame6.place(relx = 0.8, rely = 0.8, relwidth = 0.2, relheight = 0.15, anchor ="n")

button2 = Button(frame6, text= "Clear", font=40, command = clear)
button2.place(relwidth = 1, relheight = 1)

root.mainloop()


