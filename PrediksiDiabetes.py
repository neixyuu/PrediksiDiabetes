import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import threading
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns


class SistemPrediksiDiabetes:
    def __init__(self, root):
        self.root = root
        self.root.title("Prediksi Risiko Diabetes")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f5f5f5')
        
        self.dataset = None
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.model_trained = False
        
        self.label_mapping = {
            'Pregnancies': 'Jumlah Kehamilan',
            'Glucose': 'Kadar Glukosa',
            'BloodPressure': 'Tekanan Darah',
            'SkinThickness': 'Ketebalan Kulit',
            'Insulin': 'Kadar Insulin',
            'BMI': 'Indeks Massa Tubuh (BMI)',
            'DiabetesPedigreeFunction': 'Riwayat Diabetes Keluarga',
            'Age': 'Usia'
        }
        
        self.field_descriptions = {
            'Pregnancies': 'Berapa kali hamil (0 jika belum pernah / laki-laki)',
            'Glucose': 'Kadar gula darah puasa',
            'BloodPressure': 'Tekanan darah diastolik',
            'SkinThickness': 'Ketebalan lipatan kulit trisep',
            'Insulin': 'Kadar insulin serum',
            'BMI': 'Indeks massa tubuh',
            'DiabetesPedigreeFunction': 'Skor riwayat diabetes keluarga',
            'Age': 'Usia dalam tahun'
        }
        
        self.field_units = {
            'Pregnancies': 'kali',
            'Glucose': 'mg/dL',
            'BloodPressure': 'mmHg',
            'SkinThickness': 'mm',
            'Insulin': 'μU/mL',
            'BMI': 'kg/m²',
            'DiabetesPedigreeFunction': 'skor',
            'Age': 'tahun'
        }
        
        self.normal_ranges = {
            'Pregnancies': {'min': 0, 'max': 5, 'normal': '0-3', 'example': '2'},
            'Glucose': {'min': 70, 'max': 125, 'normal': '70-100', 'example': '95'},
            'BloodPressure': {'min': 60, 'max': 90, 'normal': '70-80', 'example': '75'},
            'SkinThickness': {'min': 10, 'max': 40, 'normal': '15-30', 'example': '23'},
            'Insulin': {'min': 15, 'max': 200, 'normal': '16-166', 'example': '80'},
            'BMI': {'min': 18.5, 'max': 30, 'normal': '18.5-24.9', 'example': '22.5'},
            'DiabetesPedigreeFunction': {'min': 0.0, 'max': 1.5, 'normal': '0.0-0.5', 'example': '0.3'},
            'Age': {'min': 21, 'max': 65, 'normal': '21-60', 'example': '35'}
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        self.create_header()
        self.create_dataset_panel()
        self.create_main_panels()
        
    def create_header(self):
        header_frame = tk.Frame(self.root, bg='#1e5f8e', height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title = tk.Label(header_frame, 
                        text="PREDIKSI RISIKO DIABETES",
                        font=('Arial', 24, 'bold'), 
                        fg='white', 
                        bg='#1e5f8e')
        title.pack(pady=25)
        
    def create_dataset_panel(self):
        dataset_frame = tk.Frame(self.root, bg='white', relief=tk.FLAT, borderwidth=0)
        dataset_frame.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        left_section = tk.Frame(dataset_frame, bg='white')
        left_section.pack(side=tk.LEFT, padx=15, pady=12)
        
        tk.Label(left_section, text="Dataset:", font=('Arial', 10, 'bold'), bg='white').pack(side=tk.LEFT, padx=(0, 10))
        
        self.dataset_info = tk.Label(left_section, text="Belum ada dataset", font=('Arial', 10), fg='#666', bg='white')
        self.dataset_info.pack(side=tk.LEFT)
        
        right_section = tk.Frame(dataset_frame, bg='white')
        right_section.pack(side=tk.RIGHT, padx=15, pady=12)
        
        tk.Button(right_section,
                 text="Pilih Dataset",
                 command=self.load_dataset,
                 bg='#4a90e2',
                 fg='white',
                 font=('Arial', 9, 'bold'),
                 padx=20,
                 pady=6,
                 cursor='hand2').pack(side=tk.LEFT, padx=5)
        
        self.train_button = tk.Button(right_section,
                                     text="Training Model",
                                     command=self.train_model_thread,
                                     bg='#5cb85c',
                                     fg='white',
                                     font=('Arial', 9, 'bold'),
                                     padx=20,
                                     pady=6,
                                     state='disabled',
                                     cursor='hand2')
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(right_section, text="", font=('Arial', 9), bg='white', fg='#666')
        self.status_label.pack(side=tk.LEFT, padx=15)
        
    def create_main_panels(self):
        main_container = tk.Frame(self.root, bg='#f5f5f5')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        left_panel = tk.Frame(main_container, bg='white', relief=tk.FLAT, borderwidth=0)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        panel_header = tk.Frame(left_panel, bg='#f8f9fa', height=50)
        panel_header.pack(fill=tk.X)
        panel_header.pack_propagate(False)
        
        tk.Label(panel_header,
                text="Form Prediksi Risiko Diabetes",
                font=('Arial', 13, 'bold'),
                bg='#f8f9fa').pack(pady=15)
        
        canvas = tk.Canvas(left_panel, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.input_frame = scrollable_frame
        canvas.pack(side="left", fill="both", expand=True, padx=15, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        self.entries = {}
        self.create_placeholder_form()
        
        button_container = tk.Frame(left_panel, bg='white')
        button_container.pack(pady=15)
        
        self.predict_button = tk.Button(button_container,
                                      text="PREDIKSI RISIKO",
                                      command=self.predict,
                                      bg='#ff6b35',
                                      fg='white',
                                      font=('Arial', 11, 'bold'),
                                      padx=30,
                                      pady=10,
                                      state='disabled',
                                      cursor='hand2')
        self.predict_button.pack(side=tk.LEFT, padx=8)
        
        self.reset_button = tk.Button(button_container,
                                     text="RESET FORM",
                                     command=self.reset_form,
                                     bg='#6c757d',
                                     fg='white',
                                     font=('Arial', 11, 'bold'),
                                     padx=30,
                                     pady=10,
                                     state='disabled',
                                     cursor='hand2')
        self.reset_button.pack(side=tk.LEFT, padx=8)
        
        right_panel = tk.Frame(main_container, bg='white', relief=tk.FLAT, borderwidth=0)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        result_header = tk.Frame(right_panel, bg='#f8f9fa', height=50)
        result_header.pack(fill=tk.X)
        result_header.pack_propagate(False)
        
        tk.Label(result_header,
                text="Hasil Prediksi",
                font=('Arial', 13, 'bold'),
                bg='#f8f9fa').pack(pady=15)
        
        self.model_info_frame = tk.Frame(right_panel, bg='#e8f4fd')
        self.model_info_frame.pack(fill=tk.X, padx=15, pady=10)
        self.show_model_info()
        
        #bagian confusion matrix
        cm_button_frame = tk.Frame(right_panel, bg='white')
        cm_button_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        self.cm_button = tk.Button(cm_button_frame,
                                  text="Tampilkan Confusion Matrix",
                                  command=self.show_confusion_matrix,
                                  bg='#17a2b8',
                                  fg='white',
                                  font=('Arial', 10, 'bold'),
                                  padx=20,
                                  pady=8,
                                  state='disabled',
                                  cursor='hand2')
        self.cm_button.pack()
        
        result_container = tk.Frame(right_panel, bg='white')
        result_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        self.result_display = tk.Text(result_container,
                                     wrap=tk.WORD,
                                     font=('Consolas', 9),
                                     bg='#fafafa',
                                     relief=tk.FLAT,
                                     borderwidth=0,
                                     padx=10,
                                     pady=10)
        self.result_display.pack(fill=tk.BOTH, expand=True)
        
        result_scrollbar = ttk.Scrollbar(self.result_display)
        result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_display.config(yscrollcommand=result_scrollbar.set)
        result_scrollbar.config(command=self.result_display.yview)
        
        self.show_welcome_message()
        
    def create_placeholder_form(self):
        info_label = tk.Label(self.input_frame,
                            text="Silakan pilih dataset dan training model terlebih dahulu",
                            font=('Arial', 10),
                            fg='#999',
                            bg='white')
        info_label.pack(pady=80)
        
        # bagian dimana dataset di load
    def load_dataset(self):
        file_path = filedialog.askopenfilename(
            title="Pilih File Dataset CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.dataset = pd.read_csv(file_path)
                
                if 'Outcome' not in self.dataset.columns:
                    messagebox.showerror("Error", "Dataset harus memiliki kolom 'Outcome' untuk target prediksi")
                    return
                
                self.feature_columns = [col for col in self.dataset.columns if col != 'Outcome']
                
                filename = os.path.basename(file_path)
                data_info = f"{filename} | {len(self.dataset)} data | {len(self.feature_columns)} fitur"
                self.dataset_info.config(text=data_info, fg='#333')
                
                outcome_counts = self.dataset['Outcome'].value_counts()
                non_diabetes = outcome_counts.get(0, 0)
                diabetes = outcome_counts.get(1, 0)
                
                # output informasi dataset
                info_message = f"Dataset Berhasil Dimuat!\n\n"
                info_message += f"File: {filename}\n"
                info_message += f"Total Data: {len(self.dataset)}\n"
                info_message += f"Jumlah Fitur: {len(self.feature_columns)}\n\n"
                info_message += f"Distribusi Data:\n"
                info_message += f"• Non-Diabetes: {non_diabetes} ({non_diabetes/len(self.dataset)*100:.1f}%)\n"
                info_message += f"• Diabetes: {diabetes} ({diabetes/len(self.dataset)*100:.1f}%)\n\n"
                info_message += f"Fitur: {', '.join(self.feature_columns)}"
                
                messagebox.showinfo("Dataset Dimuat", info_message)
                
                self.train_button.config(state='normal')
                self.status_label.config(text="Dataset siap untuk training", fg='#5cb85c')
                
                self.create_input_form()
                
            except Exception as e:
                messagebox.showerror("Error", f"Gagal membaca dataset:\n{str(e)}")
                
    def create_input_form(self):
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        
        self.entries = {}
        
        title_frame = tk.Frame(self.input_frame, bg='white')
        title_frame.pack(pady=(5, 15))
        
        tk.Label(title_frame,
                text="Masukkan Data Pasien",
                font=('Arial', 11, 'bold'),
                bg='white').pack()
        
        tk.Label(title_frame,
                text="Isi semua field dengan data medis",
                font=('Arial', 9),
                fg='#666',
                bg='white').pack()
        
        for column in self.feature_columns:
            field_frame = tk.Frame(self.input_frame, bg='#f8f9fa', relief=tk.FLAT)
            field_frame.pack(fill=tk.X, padx=5, pady=6)
            
            label_container = tk.Frame(field_frame, bg='#f8f9fa')
            label_container.pack(fill=tk.X, padx=12, pady=(8, 4))
            
            label_text = self.label_mapping.get(column, column)
            
            main_label = tk.Label(label_container,
                                 text=f"{label_text}",
                                 font=('Arial', 10, 'bold'),
                                 bg='#f8f9fa',
                                 anchor='w')
            main_label.pack(side=tk.TOP, anchor='w')
            
            if column in self.field_descriptions:
                desc_label = tk.Label(label_container,
                                     text=self.field_descriptions[column],
                                     font=('Arial', 8),
                                     fg='#666',
                                     bg='#f8f9fa',
                                     anchor='w')
                desc_label.pack(side=tk.TOP, anchor='w')
            
            input_container = tk.Frame(field_frame, bg='#f8f9fa')
            input_container.pack(fill=tk.X, padx=12, pady=(0, 8))
            
            entry = tk.Entry(input_container,
                           width=15,
                           font=('Arial', 10),
                           relief=tk.SOLID,
                           borderwidth=1)
            entry.pack(side=tk.LEFT, ipady=3)

            if column in self.field_units:
                unit_label = tk.Label(input_container,
                                     text=self.field_units[column],
                                     font=('Arial', 9),
                                     fg='#666',
                                     bg='#f8f9fa')
                unit_label.pack(side=tk.LEFT, padx=(8, 12))
            
            if column in self.normal_ranges:
                range_info = self.normal_ranges[column]
                range_text = f"Normal: {range_info['normal']}"
                
                range_label = tk.Label(input_container,
                                     text=range_text,
                                     font=('Arial', 8),
                                     fg='#28a745',
                                     bg='#f8f9fa')
                range_label.pack(side=tk.LEFT)
            
            self.entries[column] = entry
            
            if column == 'Pregnancies':
                entry.insert(0, "0")
            elif column == 'Glucose':
                hint_frame = tk.Frame(field_frame, bg='#e7f3ff')
                hint_frame.pack(fill=tk.X, padx=12, pady=(0, 8))
                tk.Label(hint_frame,
                         text="Normal <100 | Prediabetes 100-125 | Diabetes >125",
                         font=('Arial', 8),
                         fg='#004085',
                         bg='#e7f3ff').pack(anchor='w', padx=6, pady=3)
            elif column == 'BMI':
                hint_frame = tk.Frame(field_frame, bg='#e7f3ff')
                hint_frame.pack(fill=tk.X, padx=12, pady=(0, 8))
                tk.Label(hint_frame,
                         text="Rumus: Berat(kg) ÷ [Tinggi(m)]². Contoh: 70kg, 170cm → 70÷1.7²=24.2",
                         font=('Arial', 8),
                         fg='#004085',
                         bg='#e7f3ff').pack(anchor='w', padx=6, pady=3)
            elif column == 'SkinThickness':
                hint_frame = tk.Frame(field_frame, bg='#e7f3ff')
                hint_frame.pack(fill=tk.X, padx=12, pady=(0, 8))
                tk.Label(hint_frame,
                         text="Diukur di trisep (lengan atas belakang) menggunakan kaliper( Cukup di isi 29 jika tidak tahu )",
                         font=('Arial', 8),
                         fg='#004085',
                         bg='#e7f3ff').pack(anchor='w', padx=6, pady=3)
            elif column == 'DiabetesPedigreeFunction':
                hint_frame = tk.Frame(field_frame, bg='#e7f3ff')
                hint_frame.pack(fill=tk.X, padx=12, pady=(0, 8))
                tk.Label(hint_frame,
                         text="Rentang 0.0-2.5 (0=tidak ada riwayat, >1.0=riwayat kuat diabetes keluarga)",
                         font=('Arial', 8),
                         fg='#004085',
                         bg='#e7f3ff').pack(anchor='w', padx=6, pady=3)
                    
    def train_model_thread(self):
        if self.dataset is None:
            messagebox.showerror("Error", "Butuh Dataset")
            return
        
        self.status_label.config(text="Training model...", fg='#ff6b35')
        self.train_button.config(state='disabled')


        # PROSES TRAINING MODEL DI THREAD TERPISAH ( di bagian ini dimana dimulainya proses training model )
        
        # Menjalankan proses training di thread terpisah agar UI tidak hang
        thread = threading.Thread(target=self.train_model)
        thread.start()
        
    def train_model(self):
        try:
            # BAGIAN PRA-PROSES DATA: GANTI NILAI 0 DENGAN MEDIAN KOLOM
            zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            for col in zero_columns:
                if col in self.dataset.columns:
                    # Hitung median hanya dari nilai yang bukan 0
                    median_value = self.dataset[col][self.dataset[col] != 0].median()
                    # Ganti nilai 0 dengan median
                    self.dataset[col] = self.dataset[col].replace(0, median_value)
            
            # Pisahkan fitur (X) dan label/target (y)
            X = self.dataset[self.feature_columns]
            y = self.dataset['Outcome']
            
            #  BAGIAN SPLIT DATA TRAINING & TESTING 
            # (train_test_split): membagi data menjadi data latih dan data uji
            X_train, X_test, y_train, y_test = train_test_split(
                    # pemabagian 80% training dan 20% testing
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            #  BAGIAN NORMALISASI FITUR DENGAN STANDARDSCALER 
            # StandardScaler: menskalakan fitur ke median=0 dan standar=1
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)  # fit di data training
            X_test_scaled = self.scaler.transform(X_test)        # transform data testing dengan scaler yang sama
            
            #  BAGIAN PENCARIAN K OPTIMAL (GRIDSEARCHCV UNTUK KNN) 
            param_grid = {'n_neighbors': range(3, 31, 2)}  # coba K ganjil 3,5,...,29
            knn = KNeighborsClassifier()                   # inisialisasi model KNN
            grid_search = GridSearchCV(
                knn,
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            # Proses cross-validation untuk mencari kombinasi K terbaik
            grid_search.fit(X_train_scaled, y_train)
            
            # Simpan nilai K terbaik dan skor CV terbaik
            self.k_optimal = grid_search.best_params_['n_neighbors']
            self.best_cv_score = grid_search.best_score_
            
            #  BAGIAN TRAINING MODEL KNN DENGAN K TERBAIK 
            self.model = KNeighborsClassifier(n_neighbors=self.k_optimal)
            # fit(): proses training KNN pada data training yang sudah distandarisasi
            self.model.fit(X_train_scaled, y_train)
            
            #  BAGIAN TESTING / EVALUASI MODEL KNN 
            # predict(): menghasilkan label prediksi untuk data uji
            y_pred = self.model.predict(X_test_scaled)
            
            # Hitung metrik evaluasi (akurasi, precision, recall, f1-score)
            self.accuracy = accuracy_score(y_test, y_pred)
            self.precision = precision_score(y_test, y_pred)
            self.recall = recall_score(y_test, y_pred)
            self.f1 = f1_score(y_test, y_pred)
            
            # Confusion matrix untuk mendapatkan TP, TN, FP, FN
            cm = confusion_matrix(y_test, y_pred)
            self.cm = cm  # fungsi untuk menyimpan confusion matrix
            self.tn, self.fp, self.fn, self.tp = cm.ravel()
            
            self.model_trained = True
            
            self.root.after(0, self.on_training_complete)
        
        # fungsi jika terjadi error selama training
        except Exception as e:
            self.root.after(0, lambda: self.on_training_error(str(e)))
         
    # info setelah training selesai
    def on_training_complete(self):
        self.status_label.config(text=f"Model berhasil (Akurasi: {self.accuracy:.1%})", fg='#28a745')
        self.train_button.config(state='normal')
        self.predict_button.config(state='normal')
        self.reset_button.config(state='normal')
        self.cm_button.config(state='normal')
        
        self.show_model_info()
        
        message = f"Training Model Berhasil!\n\n"
        message += f"Parameter Optimal:\n"
        message += f"• K = {self.k_optimal}\n"
        message += f"• Cross-validation Score = {self.best_cv_score:.1%}\n\n"
        message += f"Evaluasi Model:\n"
        message += f"• Akurasi = {self.accuracy:.1%}\n"
        message += f"• Precision = {self.precision:.1%}\n"
        message += f"• Recall = {self.recall:.1%}\n"
        message += f"• F1-Score = {self.f1:.1%}\n\n"
        message += f"Model siap untuk prediksi!"
        
        messagebox.showinfo("Training Berhasil", message)
        
    def on_training_error(self, error_msg):
        self.status_label.config(text="Training gagal", fg='#dc3545')
        self.train_button.config(state='normal')
        messagebox.showerror("Error Training", f"Gagal melatih model:\n{error_msg}")
        
    def show_model_info(self):
        for widget in self.model_info_frame.winfo_children():
            widget.destroy()
        
        if not self.model_trained:
            tk.Label(self.model_info_frame,
                   text="Model belum ditraining",
                   font=('Arial', 9),
                   fg='#999',
                   bg='#e8f4fd').pack(pady=12)
        else:
            tk.Label(self.model_info_frame,
                   text="Model Information",
                   font=('Arial', 10, 'bold'),
                   bg='#e8f4fd').pack(pady=(8, 5))
            
            info_grid = tk.Frame(self.model_info_frame, bg='#e8f4fd')
            info_grid.pack(pady=(0, 8))
            
            # fungsi untuk menampilkan info model
            metrics = [
                ("K-Optimal", f"{self.k_optimal}"),
                ("Accuracy", f"{self.accuracy:.1%}"),
                ("Precision", f"{self.precision:.1%}"),
                ("Recall", f"{self.recall:.1%}"),
                ("F1-Score", f"{self.f1:.1%}"),
                ("CV-Score", f"{self.best_cv_score:.1%}")
            ]
            
            for i, (label, value) in enumerate(metrics):
                row = i // 3
                col = i % 3
                
                frame = tk.Frame(info_grid, bg='#e8f4fd')
                frame.grid(row=row, column=col, padx=15, pady=2)
                
                tk.Label(frame,
                       text=f"{label}:",
                       font=('Arial', 8),
                       fg='#555',
                       bg='#e8f4fd').pack(side=tk.LEFT)
                
                tk.Label(frame,
                       text=value,
                       font=('Arial', 8, 'bold'),
                       bg='#e8f4fd').pack(side=tk.LEFT, padx=(4, 0))
                        
    def predict(self):
        if not self.model_trained:
            messagebox.showerror("Error", "Model belum ditraining. Silakan training model terlebih dahulu.")
            return
        
        try:
            input_data = {}
            for column, entry in self.entries.items():
                value = entry.get().strip()
                if not value:
                    label = self.label_mapping.get(column, column)
                    messagebox.showerror("Error", f"Harap isi field {label}")
                    return
                input_data[column] = float(value)
            
            # Buat DataFrame 1 baris dari input user
            df_input = pd.DataFrame([input_data])
            #  SCALING DATA INPUT DENGAN SCALER YANG SAMA DARI TRAINING 
            X_scaled = self.scaler.transform(df_input)
            
            #  EKSEKUSI KNN UNTUK PREDIKSI SATU PASIEN 
            # predict() -> kelas 0/1 (tidak berisiko / berisiko)
            prediction = self.model.predict(X_scaled)[0]
            # predict_proba() -> probabilitas masing-masing kelas [P(0), P(1)]
            probability = self.model.predict_proba(X_scaled)[0]
            
            self.display_prediction_result(input_data, prediction, probability)
            
        except ValueError as e:
            messagebox.showerror("Error", f"Input tidak valid. Pastikan semua input berupa angka.\n{str(e)}")
            
    def display_prediction_result(self, input_data, prediction, probability):
        self.result_display.delete(1.0, tk.END)
        
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        
        self.result_display.insert(tk.END, "="*65 + "\n")
        self.result_display.insert(tk.END, "           HASIL PREDIKSI RISIKO DIABETES\n")
        self.result_display.insert(tk.END, "="*65 + "\n\n")
        
        self.result_display.insert(tk.END, f"Waktu Prediksi: {current_time}\n")
        self.result_display.insert(tk.END, f"Model: K-Nearest Neighbors (K={self.k_optimal})\n\n")
        
        self.result_display.insert(tk.END, "-"*65 + "\n")
        self.result_display.insert(tk.END, "DATA PASIEN:\n")
        self.result_display.insert(tk.END, "-"*65 + "\n")
        
        for column, value in input_data.items():
            label = self.label_mapping.get(column, column)
            unit = self.field_units.get(column, '')
            self.result_display.insert(tk.END, f"{label:32s}: {value:>8.2f} {unit}\n")
        
        self.result_display.insert(tk.END, "\n" + "-"*65 + "\n")
        self.result_display.insert(tk.END, "HASIL PREDIKSI:\n")
        self.result_display.insert(tk.END, "-"*65 + "\n\n")
        
        #  PERHITUNGAN PROBABILITAS RISIKO DARI OUTPUT KNN 
        # probability[1] = probabilitas kelas "berisiko" (Outcome=1)
        risk_prob = probability[1] * 100
        
        if prediction == 1:
            status = "BERISIKO DIABETES"
        else:
            status = "TIDAK BERISIKO DIABETES"
            
        self.result_display.insert(tk.END, f"Status: {status}\n\n")
        
        self.result_display.insert(tk.END, "PROBABILITAS RISIKO:\n")
        self.result_display.insert(tk.END, f"• Tidak Berisiko : {probability[0]*100:>6.2f}%\n")
        self.result_display.insert(tk.END, f"• Berisiko       : {probability[1]*100:>6.2f}%\n\n")
        
        #  VISUALISASI BAR RISIKO BERBASIS PERSENTASE 
        bar_length = 50
        filled = int(risk_prob / 2)  # 100% -> 50 blok penuh
        bar = "█" * filled + "░" * (bar_length - filled)
        
        self.result_display.insert(tk.END, "VISUALISASI RISIKO:\n")
        self.result_display.insert(tk.END, f"[{bar}] {risk_prob:.1f}%\n\n")
        
        #  LOGIKA KATEGORI RISIKO BERDASARKAN PERSEN 
        self.result_display.insert(tk.END, "KATEGORI RISIKO:\n")
        
        if risk_prob >= 75:
            kategori = "SANGAT TINGGI"
            warna = "Merah"
        elif risk_prob >= 50:
            kategori = "TINGGI"
            warna = "Orange"
        elif risk_prob >= 25:
            kategori = "SEDANG"
            warna = "Kuning"
        else:
            kategori = "RENDAH"
            warna = "Hijau"
            
        self.result_display.insert(tk.END, f"Tingkat Risiko: {kategori} ({warna})\n\n")
        
        self.result_display.insert(tk.END, "-"*65 + "\n")
        self.result_display.insert(tk.END, "INTERPRETASI & REKOMENDASI:\n")
        self.result_display.insert(tk.END, "-"*65 + "\n\n")
        
        # Rekomendasi berbasis kategori risiko
        if risk_prob >= 75:
            interpretasi = """RISIKO SANGAT TINGGI:
• Segera konsultasi dengan dokter spesialis
• Pemeriksaan gula darah komprehensif mendesak
• Perubahan gaya hidup harus segera dimulai
• Monitor gula darah rutin (harian)"""
        elif risk_prob >= 50:
            interpretasi = """RISIKO TINGGI:
• Konsultasi dengan dokter dalam waktu dekat
• Lakukan tes HbA1c dan glukosa puasa
• Mulai program diet dan olahraga
• Monitor gula darah mingguan"""
        elif risk_prob >= 25:
            interpretasi = """RISIKO SEDANG:
• Pemeriksaan kesehatan rutin tiap 3-6 bulan
• Perhatikan pola makan dan aktivitas fisik
• Monitor berat badan dan BMI
• Tes gula darah berkala"""
        else:
            interpretasi = """RISIKO RENDAH:
• Pertahankan gaya hidup sehat
• Pemeriksaan tahunan tetap dianjurkan
• Jaga berat badan ideal
• Olahraga teratur min 150 menit/minggu"""
            
        self.result_display.insert(tk.END, interpretasi + "\n\n")
        
        self.result_display.insert(tk.END, "-"*65 + "\n")
        self.result_display.insert(tk.END, "FAKTOR RISIKO PERLU DIPERHATIKAN:\n")
        self.result_display.insert(tk.END, "-"*65 + "\n")
        
        #  LOGIKA CEK FAKTOR RISIKO BERDASARKAN NILAI INPUT 
        warnings = []
        if 'BMI' in input_data and input_data['BMI'] > 25:
            warnings.append("• BMI tinggi - pertimbangkan penurunan berat badan")
        if 'Glucose' in input_data and input_data['Glucose'] > 100:
            warnings.append("• Glukosa tinggi - perlu monitoring ketat")
        if 'Age' in input_data and input_data['Age'] > 45:
            warnings.append("• Usia >45 tahun - risiko meningkat")
        if 'BloodPressure' in input_data and input_data['BloodPressure'] > 80:
            warnings.append("• Tekanan darah perlu diperhatikan")
        if 'Pregnancies' in input_data and input_data['Pregnancies'] > 3:
            warnings.append("• Riwayat kehamilan banyak - risiko lebih tinggi")
            
        if warnings:
            for w in warnings:
                self.result_display.insert(tk.END, w + "\n")
        else:
            self.result_display.insert(tk.END, "• Tidak ada faktor risiko signifikan terdeteksi\n")
            
        self.result_display.insert(tk.END, "\n" + "="*65 + "\n")
        self.result_display.insert(tk.END, "CATATAN:\n")
        self.result_display.insert(tk.END, "Hasil prediksi ini berdasarkan model machine learning\n")
        self.result_display.insert(tk.END, "dan BUKAN diagnosis medis definitif. Konsultasikan\n")
        self.result_display.insert(tk.END, "dengan tenaga medis untuk diagnosis akurat.\n")
        self.result_display.insert(tk.END, "="*65 + "\n")
        
    def show_welcome_message(self):
        self.result_display.insert(tk.END, "="*65 + "\n")
        self.result_display.insert(tk.END, "     SELAMAT DATANG DI SISTEM PREDIKSI DIABETES\n")
        self.result_display.insert(tk.END, "="*65 + "\n\n")
        
        self.result_display.insert(tk.END, "Sistem ini menggunakan algoritma K-Nearest Neighbors\n")
        self.result_display.insert(tk.END, "untuk prediksi risiko diabetes berdasarkan data klinis.\n\n")
        
        self.result_display.insert(tk.END, "CARA PENGGUNAAN:\n")
        self.result_display.insert(tk.END, "-"*45 + "\n")
        self.result_display.insert(tk.END, "1. Klik 'Pilih Dataset' untuk memuat data CSV\n")
        self.result_display.insert(tk.END, "2. Klik 'Training Model' untuk melatih model\n")
        self.result_display.insert(tk.END, "3. Isi form dengan data pasien\n")
        self.result_display.insert(tk.END, "4. Klik 'PREDIKSI RISIKO' untuk hasil\n")
        self.result_display.insert(tk.END, "5. Gunakan 'RESET FORM' untuk input baru\n\n")
        
        self.result_display.insert(tk.END, "="*65 + "\n")
        

    # fungsi untuk mereset form input
    def reset_form(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        
        if 'Pregnancies' in self.entries:
            self.entries['Pregnancies'].insert(0, "0")
        
        self.show_welcome_message()
    
    def show_confusion_matrix(self):
        if not self.model_trained or not hasattr(self, 'cm'):
            messagebox.showerror("Error", "Model belum ditraining. Silakan training model terlebih dahulu.")
            return
        
        # bagian untuk menampilkan confusion matrix
        cm_window = tk.Toplevel(self.root)
        cm_window.title("Confusion Matrix - Hasil Evaluasi Model")
        cm_window.geometry("700x600")
        cm_window.configure(bg='white')
        
        header_frame = tk.Frame(cm_window, bg='#1e5f8e', height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame,
                text="CONFUSION MATRIX",
                font=('Arial', 18, 'bold'),
                fg='white',
                bg='#1e5f8e').pack(pady=18)
        
        # Frame untuk matplotlib
        plot_frame = tk.Frame(cm_window, bg='white')
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Buat figure matplotlib
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # Buat annotasi dengan label TN, TP, FN, FP untuk setiap sel
        # Format: [baris, kolom] = [aktual, prediksi]
        # [0,0] = TN, [0,1] = FP, [1,0] = FN, [1,1] = TP
        annot_labels = [
            [f'True Negative \n( Tidak Diabetes )\n{self.tn}', f'False Positive \n ( Diabetes )\n{self.fp}'],
            [f'False Negative \n ( Tidak Diabetes )\n{self.fn}', f'True Positive \n ( Diabetes )\n{self.tp}']
        ]
        
        # Visualisasi confusion matrix dengan seaborn
        sns.heatmap(self.cm, annot=annot_labels, fmt='', cmap='Blues', 
                #    xticklabels=['Tidak Diabetes', 'Diabetes'],
                #    yticklabels=['Tidak Diabetes', 'Diabetes'],
                   ax=ax, cbar_kws={'label': 'Jumlah'})
        
        ax.set_xlabel('Prediksi', fontsize=12, fontweight='bold')
        ax.set_ylabel('Aktual', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix - Model KNN', fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        # Embed matplotlib figure ke tkinter
        canvas = FigureCanvasTkAgg(fig, plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Frame untuk informasi detail
        info_frame = tk.Frame(cm_window, bg='#f8f9fa', relief=tk.FLAT)
        info_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Informasi detail confusion matrix
        info_text = f"""
        True Negative (TN): {self.tn}    |    False Positive (FP): {self.fp}
        False Negative (FN): {self.fn}   |    True Positive (TP): {self.tp}
        
        Akurasi: {self.accuracy:.2%}    |    Precision: {self.precision:.2%}
        Recall: {self.recall:.2%}       |    F1-Score: {self.f1:.2%}
        """
        
        tk.Label(info_frame,
                text=info_text,
                font=('Consolas', 9),
                bg='#f8f9fa',
                justify=tk.LEFT,
                anchor='w').pack(padx=15, pady=10)
        
        # bagian tambahan untuk penjelasan
        explanation_frame = tk.Frame(cm_window, bg='white')
        explanation_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        explanation = """
        Keterangan:
        • True Negative (TN): Benar memprediksi tidak diabetes
        • False Positive (FP): Salah memprediksi diabetes (padahal tidak)
        • False Negative (FN): Salah memprediksi tidak diabetes (padahal diabetes)
        • True Positive (TP): Benar memprediksi diabetes
        """
        
        tk.Label(explanation_frame,
                text=explanation,
                font=('Arial', 8),
                bg='white',
                fg='#666',
                justify=tk.LEFT,
                anchor='w').pack(padx=10)
        
def main():
    root = tk.Tk()
    app = SistemPrediksiDiabetes(root)
    root.mainloop()


if __name__ == "__main__":
    main()
