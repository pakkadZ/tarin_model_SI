import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Combobox, Progressbar, Style
import threading
import os
from ultralytics import YOLO

class YOLOTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 YOLO Training GUI")
        self.root.geometry("700x650")
        self.root.configure(bg="#f0f0f0")

        self.font_large = ("Helvetica", 14)
        self.font_medium = ("Helvetica", 12)
        self.pad_y = 8

        style = Style()
        style.configure("TButton", font=self.font_medium, padding=6)
        style.configure("TCombobox", font=self.font_medium)
        style.configure("TProgressbar", thickness=20)

        # -------- Dataset Path --------
        tk.Label(root, text="📂 Dataset Path (.yaml):", font=self.font_large, bg="#f0f0f0").pack(pady=(15, 5))
        self.dataset_path_var = tk.StringVar()
        tk.Entry(root, textvariable=self.dataset_path_var, width=60, font=self.font_medium).pack()
        tk.Button(root, text="Browse", font=self.font_medium, command=self.browse_dataset, height=1, width=10).pack(pady=self.pad_y)

        # -------- Model Selection --------
        tk.Label(root, text="🧠 YOLO Model (.pt):", font=self.font_large, bg="#f0f0f0").pack(pady=(20, 5))
        self.model_var = tk.StringVar()
        self.model_combobox = Combobox(root, textvariable=self.model_var,
                                       values=["yolo11n.pt", "yolov8n.pt", "custom.pt"], state='readonly', width=40)
        self.model_combobox.current(0)
        self.model_combobox.pack(pady=self.pad_y)

        # -------- Epochs --------
        tk.Label(root, text="⏱️ Epochs:", font=self.font_large, bg="#f0f0f0").pack(pady=(20, 5))
        self.epochs_var = tk.IntVar(value=100)
        tk.Entry(root, textvariable=self.epochs_var, font=self.font_medium, width=10).pack()

        # -------- Device Selection --------
        tk.Label(root, text="⚙️ Device:", font=self.font_large, bg="#f0f0f0").pack(pady=(20, 5))
        self.device_var = tk.StringVar()
        self.device_combobox = Combobox(root, textvariable=self.device_var,
                                        values=["cpu", "0", "1", "2", "3"], state='readonly', width=10)
        self.device_combobox.current(1)
        self.device_combobox.pack(pady=self.pad_y)

        # -------- Buttons --------
        tk.Button(root, text="✅ Confirm Setup", font=self.font_medium, width=25, command=self.confirm_setup).pack(pady=(25, 5))

        self.train_button = tk.Button(root, text="🚀 Start Training", font=self.font_medium, width=25,
                                      command=self.start_training, state='disabled')
        self.train_button.pack(pady=self.pad_y)

        self.show_folder_button = tk.Button(root, text="📂 Open Model Folder", font=self.font_medium, width=25,
                                            command=self.open_model_folder, state='disabled')
        self.show_folder_button.pack(pady=self.pad_y)

        # -------- Progress Bar --------
        self.progress = Progressbar(root, orient=tk.HORIZONTAL, length=400, mode='indeterminate')
        self.progress.pack(pady=25)

        self.output_folder = ""

    def browse_dataset(self):
        initial_dir = r"C:\Users\pakka\OneDrive\Desktop\train_model\dataset_FLODER"
        path = filedialog.askopenfilename(initialdir=initial_dir, filetypes=[("YAML Files", "*.yaml")])
        if path:
            self.dataset_path_var.set(path)

    def confirm_setup(self):
        path = self.dataset_path_var.get()
        if not os.path.isfile(path) or not path.endswith('.yaml'):
            messagebox.showerror("Error", "❌ Please provide a valid .yaml file.")
            return
        self.train_button.config(state='normal')
        messagebox.showinfo("Setup", "✅ Setup confirmed. Ready to train!")

    def start_training(self):
        self.progress.start()
        self.train_button.config(state='disabled')
        self.show_folder_button.config(state='disabled')
        threading.Thread(target=self.train_model).start()

    def train_model(self):
        try:
            model = YOLO(self.model_var.get())
            results = model.train(
                data=self.dataset_path_var.get(),
                epochs=self.epochs_var.get(),
                imgsz=640,
                device=self.device_var.get()
            )
            self.output_folder = results.save_dir if hasattr(results, 'save_dir') else "runs/train/exp"
            self.progress.stop()
            self.show_folder_button.config(state='normal')
            messagebox.showinfo("Training", "✅ Training Completed!")
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"❌ Error occurred:\n{str(e)}")

    def open_model_folder(self):
        if self.output_folder and os.path.exists(self.output_folder):
            os.startfile(self.output_folder)
        else:
            fallback_path = os.path.join("runs", "train", "exp")
            if os.path.exists(fallback_path):
                os.startfile(fallback_path)
            else:
                messagebox.showerror("Error", "❌ Model folder not found.")

if __name__ == '__main__':
    root = tk.Tk()
    app = YOLOTrainerApp(root)
    root.mainloop()
