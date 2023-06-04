#%%
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import numpy as np
from validate import predict
#%%

class Window(tk.Tk):
    def __init__(self, screenName: str | None = None, baseName: str | None = None, className: str = "Tk", useTk: bool = True, sync: bool = False, use: str | None = None) -> None:
        super().__init__(screenName, baseName, className, useTk, sync, use)
        #self.geometry("400x700")
        self.title('Dish Classification')
        self.font = ('times', 18, 'bold')
        self.image = Image.fromarray(np.ones((224, 224, 3), 'uint8'), )
        self.image_tk = ImageTk.PhotoImage(self.image)
        b1 = tk.Button(self, text='Upload File',
                    width=20,command = self.upload_file)
        b1.pack()
        
        self.b2 = tk.Button(self, image=self.image_tk)
        self.b2.pack(pady = 15)
        self.b3 = tk.Button(self, text='Predict', command=self.predict_image)
        self.b3.pack(pady = 15)
        self.l1 = tk.Label(self, text='Predicted class: ---')
        self.l1.pack(pady = 15)
        
        
    
    def upload_file(self):
        f_types = [('Jpg Files', '*.jpg'), ('Png Files', '*.png')]
        filename = filedialog.askopenfilename(filetypes=f_types, initialdir='school_lunch/cropped')
        self.image = Image.open(filename)
        self.image = self.image.resize((224,224))
        self.image_tk = ImageTk.PhotoImage(self.image)
        self.b2.configure(image=self.image_tk)
    
    def predict_image(self):
        class_ = predict(self.image)
        print(class_)
        self.l1.configure(text=f'Predicted class: {class_}')
        

# %%
app = Window()
app.mainloop()
