import os
import project
import numpy as np
from pickle import load
from PIL import Image, ImageTk
from tkinter import Tk, Entry, Label, Button, Frame, filedialog, LEFT, RIGHT, BOTTOM, END

model_path = 'D:/FCIS - ASU/Y4S2/Human Computer Interface/Project/svm_model.pkl'

class PhotoViewer:
    def __init__(self, root):
        self.root = root
        self.folder = "D:/FCIS - ASU/Y4S2/Human Computer Interface/Project/images"
        self.images = [f for f in os.listdir(self.folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.images.sort()
        self.index = 0

        self.main_frame = Frame(root)
        self.main_frame.pack()

        self.prev_label = Label(self.main_frame)
        self.prev_label.pack(side=LEFT, padx=10)

        self.current_label = Label(self.main_frame)
        self.current_label.pack(side=LEFT, padx=10)

        self.next_label = Label(self.main_frame)
        self.next_label.pack(side=LEFT, padx=10)

        self.button_frame = Frame(root)
        self.button_frame.pack(side=BOTTOM, pady=10)

        self.prev_button = Button(self.button_frame, text="<< Previous", font=("Arial", 12), command=self.show_previous, width=12, height=2)
        self.prev_button.pack(side=LEFT, padx=20, pady=10)

        self.next_button = Button(self.button_frame, text="Next >>", font=("Arial", 12), command=self.show_next, width=12, height=2)
        self.next_button.pack(side=RIGHT, padx=20, pady=10)

        frame1 = Frame(root)
        frame1.pack(padx=10)

        self.entry1 = Entry(frame1, width=60, font=("Arial", 12))
        self.entry1.pack(side=LEFT, padx=(0, 10))

        self.button1 = Button(frame1, text="Upload Signal 1", font=("Arial", 12), command=self.upload_file1)
        self.button1.pack(side=LEFT)

        frame2 = Frame(root)
        frame2.pack(padx=10)

        self.entry2 = Entry(frame2, width=60, font=("Arial", 12))
        self.entry2.pack(side=LEFT, padx=(0, 10))

        self.button2 = Button(frame2, text="Upload Signal 2", font=("Arial", 12), command=self.upload_file2)
        self.button2.pack(side=LEFT)

        frame3 = Frame(root)
        frame3.pack(padx=10)

        self.ok_button = Button(frame3, text="OK", font=("Arial", 12), command=self.navigate, width=12, height=2)
        self.ok_button.pack(side=LEFT, padx=20, pady=10)

        self.update_images()

    def load_image(self, filename, size=(250, 250)):
        img_path = os.path.join(self.folder, filename)
        img = Image.open(img_path)
        img.thumbnail(size)
        return ImageTk.PhotoImage(img)

    def update_images(self):
        total = len(self.images)
        prev_index = (self.index - 1) % total
        next_index = (self.index + 1) % total

        self.prev_img = self.load_image(self.images[prev_index])
        self.curr_img = self.load_image(self.images[self.index], size=(550, 550))
        self.next_img = self.load_image(self.images[next_index])

        self.prev_label.configure(image=self.prev_img)
        self.current_label.configure(image=self.curr_img)
        self.next_label.configure(image=self.next_img)

    def show_next(self):
        self.index = (self.index + 1) % len(self.images)
        self.update_images()

    def show_previous(self):
        self.index = (self.index - 1) % len(self.images)
        self.update_images()

    def upload_file1(self):
        file_path = filedialog.askopenfilename(title="Select File 1")
        if file_path:
            self.entry1.delete(0, END)
            self.entry1.insert(0, file_path)

    def upload_file2(self):
        file_path = filedialog.askopenfilename(title="Select File 2")
        if file_path:
            self.entry2.delete(0, END)
            self.entry2.insert(0, file_path)

    def navigate(self):
        try:
            signal1 = project.read_file(str(self.entry1.get()))
            signal2 = project.read_file(str(self.entry2.get()))

            signal1 = project.bandpass_filter(signal1)
            signal2 = project.bandpass_filter(signal2)

            signal1 = project.resampling(signal1)
            signal2 = project.resampling(signal2)

            wavelet1 = project.calc_wavelet(signal1)
            wavelet2 = project.calc_wavelet(signal2)

            wavelet = np.array([wavelet1, wavelet2]).flatten()
            wavelet = wavelet[:-1]
            wavelet = [wavelet, wavelet]

            prediction = loaded_model.predict(wavelet)

            if prediction[0] == 3:
                print("Right")
                self.show_next()
            elif prediction[0] == 4:
                print("Left")
                self.show_previous()

        except FileNotFoundError:
            print("Choose two signals first")

if __name__ == "__main__":
    if not os.path.exists(model_path):
        project.main()
        
    with open(model_path, 'rb') as file:
        loaded_model = load(file)

    root = Tk()
    root.title("Photo Viewer")
    app = PhotoViewer(root)
    root.mainloop()