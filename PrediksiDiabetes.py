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

class SistemPrediksiDiabetes:
    def __init__(self, root):
        self.root = root
        self.root.title("Prediksi Risiko Diabetes")
        self.root.geometry("1300x850")
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
            'Pregnancies': 'Berapa kali hamil (0 jika belum pernah)',
            'Glucose': 'Kadar gula darah puasa (mg/dL)',
            'BloodPressure': 'Tekanan darah diastolik (mmHg)',
            'SkinThickness': 'Ketebalan lipatan kulit trisep (mm)',
            'Insulin': 'Kadar insulin 2 jam setelah makan (μU/mL)',
            'BMI': 'Berat badan (kg) / [Tinggi (m)]²',
            'DiabetesPedigreeFunction': 'Skor riwayat diabetes dalam keluarga (0.0-2.5)',
            'Age': 'Usia pasien dalam tahun'
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
        
        self.setup_ui()
        
    def setup_ui(self):
        self.create_header()
        self.create_dataset_panel()
        self.create_main_panels()
        
    def create_header(self):
        header_frame = tk.Frame(self.root, bg='#1e5f8e', height=100)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title = tk.Label(header_frame, 
                        text="PREDIKSI RISIKO DIABETES",
                        font=('Arial', 26, 'bold'), 
                        fg='white', 
                        bg='#1e5f8e')
        title.pack(pady=20)
        
    def create_dataset_panel(self):
        dataset_frame = tk.Frame(self.root, bg='white', relief=tk.RIDGE, borderwidth=1)
        dataset_frame.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        left_section = tk.Frame(dataset_frame, bg='white')
        left_section.pack(side=tk.LEFT, padx=20, pady=15)
        
        tk.Label(left_section, text="Dataset:", font=('Arial', 11, 'bold'), bg='white').pack(side=tk.LEFT, padx=(0, 10))
        
        self.dataset_info = tk.Label(left_section, text="Belum ada dataset", font=('Arial', 10), fg='#888', bg='white')
        self.dataset_info.pack(side=tk.LEFT)
        
        right_section = tk.Frame(dataset_frame, bg='white')
        right_section.pack(side=tk.RIGHT, padx=20, pady=15)
        
        tk.Button(right_section,
                 text="Pilih Dataset",
                 command=self.load_dataset,
                 bg='#4a90e2',
                 fg='white',
                 font=('Arial', 10, 'bold'),
                 padx=20,
                 pady=5).pack(side=tk.LEFT, padx=5)
        
        self.train_button = tk.Button(right_section,
                                     text="Training Model",
                                     command=self.train_model_thread,
                                     bg='#5cb85c',
                                     fg='white',
                                     font=('Arial', 10, 'bold'),
                                     padx=20,
                                     pady=5,
                                     state='disabled')
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(right_section, text="", font=('Arial', 9), bg='white')
        self.status_label.pack(side=tk.LEFT, padx=15)
        
    def create_main_panels(self):
        main_container = tk.Frame(self.root, bg='#f5f5f5')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        left_panel = tk.Frame(main_container, bg='white', relief=tk.RAISED, borderwidth=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        panel_header = tk.Frame(left_panel, bg='#f8f9fa')
        panel_header.pack(fill=tk.X)
        
        tk.Label(panel_header,
                text="Form Prediksi Risiko Diabetes",
                font=('Arial', 14, 'bold'),
                bg='#f8f9fa').pack(pady=15)
        
        canvas = tk.Canvas(left_panel, bg='white')
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.input_frame = scrollable_frame
        canvas.pack(side="left", fill="both", expand=True, padx=20)
        scrollbar.pack(side="right", fill="y")
        
        self.entries = {}
        self.create_placeholder_form()
        
        button_container = tk.Frame(left_panel, bg='white')
        button_container.pack(pady=20)
        
        self.predict_button = tk.Button(button_container,
                                      text="PREDIKSI RISIKO",
                                      command=self.predict,
                                      bg='#ff6b35',
                                      fg='white',
                                      font=('Arial', 12, 'bold'),
                                      padx=25,
                                      pady=10,
                                      state='disabled')
        self.predict_button.pack(side=tk.LEFT, padx=10)
        
        self.reset_button = tk.Button(button_container,
                                     text="RESET FORM",
                                     command=self.reset_form,
                                     bg='#dc3545',
                                     fg='white',
                                     font=('Arial', 12, 'bold'),
                                     padx=25,
                                     pady=10,
                                     state='disabled')
        self.reset_button.pack(side=tk.LEFT, padx=10)
        
        right_panel = tk.Frame(main_container, bg='white', relief=tk.RAISED, borderwidth=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        result_header = tk.Frame(right_panel, bg='#f8f9fa')
        result_header.pack(fill=tk.X)
        
        tk.Label(result_header,
                text="Hasil Prediksi",
                font=('Arial', 14, 'bold'),
                bg='#f8f9fa').pack(pady=15)
        
        self.model_info_frame = tk.Frame(right_panel, bg='#e8f4fd')
        self.model_info_frame.pack(fill=tk.X, padx=20, pady=10)
        self.show_model_info()
        
        result_container = tk.Frame(right_panel, bg='white')
        result_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.result_display = tk.Text(result_container,
                                     wrap=tk.WORD,
                                     font=('Consolas', 10),
                                     bg='#f9f9f9',
                                     relief=tk.FLAT,
                                     borderwidth=1)
        self.result_display.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(self.result_display)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_display.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.result_display.yview)
        
        self.show_welcome_message()
        
        
    def create_placeholder_form(self):
        info_label = tk.Label(self.input_frame,
                            text="Silakan pilih dataset dan training model terlebih dahulu",
                            font=('Arial', 11),
                            fg='#888',
                            bg='white')
        info_label.pack(pady=50)
        
    def load_dataset(self):
        file_path = filedialog.askopenfilename(
            title="Pilih File Dataset CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir="/mnt/user-data/uploads"
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
                self.dataset_info.config(text=data_info, fg='black')
                
                outcome_counts = self.dataset['Outcome'].value_counts()
                non_diabetes = outcome_counts.get(0, 0)
                diabetes = outcome_counts.get(1, 0)
                
                info_message = f"Dataset Berhasil Dimuat!\n\n"
                info_message += f"File: {filename}\n"
                info_message += f"Total Data: {len(self.dataset)}\n"
                info_message += f"Jumlah Fitur: {len(self.feature_columns)}\n\n"
                info_message += f"Distribusi Data:\n"
                info_message += f"• Non-Diabetes: {non_diabetes} ({non_diabetes/len(self.dataset)*100:.1f}%)\n"
                info_message += f"• Diabetes: {diabetes} ({diabetes/len(self.dataset)*100:.1f}%)\n\n"
                info_message += f"Fitur yang tersedia:\n{', '.join(self.feature_columns)}"
                
                messagebox.showinfo("Dataset Dimuat", info_message)
                
                self.train_button.config(state='normal')
                self.status_label.config(text="Dataset siap untuk training", fg='blue')
                
                self.create_input_form()
                
            except Exception as e:
                messagebox.showerror("Error", f"Gagal membaca dataset:\n{str(e)}")
                
    def create_input_form(self):
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        
        self.entries = {}
        
        title_frame = tk.Frame(self.input_frame, bg='white')
        title_frame.pack(pady=(10, 20))
        
        tk.Label(title_frame,
                text="Masukkan Data Pasien",
                font=('Arial', 13, 'bold'),
                bg='white').pack()
        
        tk.Label(title_frame,
                text="Isi semua field dengan data medis pasien",
                font=('Arial', 9),
                fg='#666',
                bg='white').pack()
        
        for column in self.feature_columns:
            field_frame = tk.Frame(self.input_frame, bg='white', relief=tk.GROOVE, borderwidth=1)
            field_frame.pack(fill=tk.X, padx=10, pady=8)
            
            label_frame = tk.Frame(field_frame, bg='white')
            label_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
            
            if column in self.label_mapping:
                label_text = self.label_mapping[column]
            else:
                label_text = column
            
            main_label = tk.Label(label_frame,
                                 text=f"{label_text}:",
                                 font=('Arial', 11, 'bold'),
                                 bg='white')
            main_label.pack(anchor='w')
            
            if column in self.field_descriptions:
                desc_label = tk.Label(label_frame,
                                    text=self.field_descriptions[column],
                                    font=('Arial', 9),
                                    fg='#555',
                                    bg='white')
                desc_label.pack(anchor='w')
            
            input_frame = tk.Frame(field_frame, bg='white')
            input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            entry = tk.Entry(input_frame,
                           width=20,
                           font=('Arial', 11),
                           relief=tk.SOLID,
                           borderwidth=1)
            entry.pack(side=tk.LEFT)
            
            if column in self.field_units:
                unit_label = tk.Label(input_frame,
                                    text=self.field_units[column],
                                    font=('Arial', 10),
                                    fg='#666',
                                    bg='white')
                unit_label.pack(side=tk.LEFT, padx=(10, 0))
            
            if column in self.dataset.columns:
                min_val = self.dataset[column].min()
                max_val = self.dataset[column].max()
                mean_val = self.dataset[column].mean()
                
                range_label = tk.Label(input_frame,
                                     text=f"(Normal: {min_val:.1f}-{max_val:.1f}, Rata-rata: {mean_val:.1f})",
                                     font=('Arial', 9),
                                     fg='#888',
                                     bg='white')
                range_label.pack(side=tk.LEFT, padx=(15, 0))
            
            self.entries[column] = entry
            
            if column == 'Pregnancies':
                entry.insert(0, "0")
            elif column == 'Glucose':
                example_frame = tk.Frame(field_frame, bg='#f0f8ff')
                example_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
                tk.Label(example_frame,
                       text="💡 Contoh: 110 (normal: <100, prediabetes: 100-125, diabetes: >125)",
                       font=('Arial', 8),
                       fg='#0066cc',
                       bg='#f0f8ff').pack(padx=5, pady=3)
            elif column == 'BMI':
                example_frame = tk.Frame(field_frame, bg='#f0f8ff')
                example_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
                tk.Label(example_frame,
                       text="💡 Cara hitung: Berat(kg)÷[Tinggi(m)×Tinggi(m)]. Contoh: 70kg, 170cm → 70÷(1.7×1.7)=24.2",
                       font=('Arial', 8),
                       fg='#0066cc',
                       bg='#f0f8ff').pack(padx=5, pady=3)
            elif column == 'DiabetesPedigreeFunction':
                example_frame = tk.Frame(field_frame, bg='#f0f8ff')
                example_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
                tk.Label(example_frame,
                       text="💡 Skor 0.0-2.5 (0=tidak ada riwayat, >1=riwayat kuat)",
                       font=('Arial', 8),
                       fg='#0066cc',
                       bg='#f0f8ff').pack(padx=5, pady=3)
                       
    def train_model_thread(self):
        if self.dataset is None:
            messagebox.showerror("Error", "Pilih dataset terlebih dahulu")
            return
        
        self.status_label.config(text="Training model...", fg='orange')
        self.train_button.config(state='disabled')
        
        thread = threading.Thread(target=self.train_model)
        thread.start()
        
    def train_model(self):
        try:
            zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            for col in zero_columns:
                if col in self.dataset.columns:
                    median_value = self.dataset[col][self.dataset[col] != 0].median()
                    self.dataset[col] = self.dataset[col].replace(0, median_value)
            
            X = self.dataset[self.feature_columns]
            y = self.dataset['Outcome']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            param_grid = {'n_neighbors': range(3, 31, 2)}
            knn = KNeighborsClassifier()
            grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            
            self.k_optimal = grid_search.best_params_['n_neighbors']
            self.best_cv_score = grid_search.best_score_
            
            self.model = KNeighborsClassifier(n_neighbors=self.k_optimal)
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            
            self.accuracy = accuracy_score(y_test, y_pred)
            self.precision = precision_score(y_test, y_pred)
            self.recall = recall_score(y_test, y_pred)
            self.f1 = f1_score(y_test, y_pred)
            
            cm = confusion_matrix(y_test, y_pred)
            self.tn, self.fp, self.fn, self.tp = cm.ravel()
            
            self.model_trained = True
            
            self.root.after(0, self.on_training_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self.on_training_error(str(e)))
            
    def on_training_complete(self):
        self.status_label.config(text=f"Model berhasil (Akurasi: {self.accuracy:.1%})", fg='green')
        self.train_button.config(state='normal')
        self.predict_button.config(state='normal')
        self.reset_button.config(state='normal')
        
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
        self.status_label.config(text="Training gagal", fg='red')
        self.train_button.config(state='normal')
        messagebox.showerror("Error Training", f"Gagal melatih model:\n{error_msg}")
        
    def show_model_info(self):
        for widget in self.model_info_frame.winfo_children():
            widget.destroy()
        
        if not self.model_trained:
            tk.Label(self.model_info_frame,
                   text="Model belum ditraining",
                   font=('Arial', 10),
                   fg='#888',
                   bg='#e8f4fd').pack(pady=15)
        else:
            tk.Label(self.model_info_frame,
                   text="Model Information",
                   font=('Arial', 11, 'bold'),
                   bg='#e8f4fd').pack(pady=(10, 5))
            
            info_grid = tk.Frame(self.model_info_frame, bg='#e8f4fd')
            info_grid.pack(pady=(0, 10))
            
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
                frame.grid(row=row, column=col, padx=10, pady=3)
                
                tk.Label(frame,
                       text=f"{label}:",
                       font=('Arial', 9),
                       bg='#e8f4fd').pack(side=tk.LEFT)
                
                tk.Label(frame,
                       text=value,
                       font=('Arial', 9, 'bold'),
                       bg='#e8f4fd').pack(side=tk.LEFT, padx=(5, 0))
                       
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
            
            df_input = pd.DataFrame([input_data])
            X_scaled = self.scaler.transform(df_input)
            
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0]
            
            neighbors = self.model.kneighbors(X_scaled, n_neighbors=self.k_optimal, return_distance=True)
            avg_distance = neighbors[0][0].mean()
            
            self.display_prediction_result(input_data, prediction, probability, avg_distance)
            
        except ValueError as e:
            messagebox.showerror("Error", f"Input tidak valid. Pastikan semua input berupa angka.\n{str(e)}")
            
    def display_prediction_result(self, input_data, prediction, probability, avg_distance):
        self.result_display.delete(1.0, tk.END)
        
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        
        self.result_display.insert(tk.END, "="*60 + "\n")
        self.result_display.insert(tk.END, "         HASIL PREDIKSI RISIKO DIABETES\n")
        self.result_display.insert(tk.END, "="*60 + "\n\n")
        
        self.result_display.insert(tk.END, f"Waktu Prediksi: {current_time}\n")
        self.result_display.insert(tk.END, f"Model: K-Nearest Neighbors (K={self.k_optimal})\n\n")
        
        self.result_display.insert(tk.END, "-"*60 + "\n")
        self.result_display.insert(tk.END, "DATA PASIEN:\n")
        self.result_display.insert(tk.END, "-"*60 + "\n")
        
        for column, value in input_data.items():
            label = self.label_mapping.get(column, column)
            unit = self.field_units.get(column, '')
            self.result_display.insert(tk.END, f"{label:30s}: {value:>10.2f} {unit}\n")
        
        self.result_display.insert(tk.END, "\n" + "-"*60 + "\n")
        self.result_display.insert(tk.END, "HASIL PREDIKSI:\n")
        self.result_display.insert(tk.END, "-"*60 + "\n\n")
        
        risk_prob = probability[1] * 100
        
        if prediction == 1:
            status = "BERISIKO DIABETES"
            status_color = "#ff4444"
        else:
            status = "TIDAK BERISIKO DIABETES"
            status_color = "#44ff44"
            
        self.result_display.insert(tk.END, f"Status: {status}\n\n")
        
        self.result_display.insert(tk.END, "PROBABILITAS RISIKO:\n")
        self.result_display.insert(tk.END, f"• Tidak Berisiko : {probability[0]*100:>6.2f}%\n")
        self.result_display.insert(tk.END, f"• Berisiko       : {probability[1]*100:>6.2f}%\n\n")
        
        bar_length = 50
        filled = int(risk_prob / 2)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        self.result_display.insert(tk.END, "VISUALISASI RISIKO:\n")
        self.result_display.insert(tk.END, f"[{bar}] {risk_prob:.1f}%\n\n")
        
        self.result_display.insert(tk.END, "KATEGORI RISIKO:\n")
        
        if risk_prob >= 75:
            kategori = "SANGAT TINGGI"
            warna_kategori = "Merah"
        elif risk_prob >= 50:
            kategori = "TINGGI"
            warna_kategori = "Orange"
        elif risk_prob >= 25:
            kategori = "SEDANG"
            warna_kategori = "Kuning"
        else:
            kategori = "RENDAH"
            warna_kategori = "Hijau"
            
        self.result_display.insert(tk.END, f"Tingkat Risiko: {kategori} ({warna_kategori})\n\n")
        
        self.result_display.insert(tk.END, "-"*60 + "\n")
        self.result_display.insert(tk.END, "INTERPRETASI & REKOMENDASI:\n")
        self.result_display.insert(tk.END, "-"*60 + "\n\n")
        
        if risk_prob >= 75:
            interpretasi = """RISIKO SANGAT TINGGI:
• Segera lakukan konsultasi dengan dokter spesialis
• Pemeriksaan gula darah komprehensif sangat mendesak
• Perubahan gaya hidup harus segera dimulai
• Monitor gula darah secara rutin (harian)
• Evaluasi komplikasi diabetes"""
        elif risk_prob >= 50:
            interpretasi = """RISIKO TINGGI:
• Konsultasi dengan dokter dalam waktu dekat
• Lakukan tes HbA1c dan glukosa puasa
• Mulai program diet dan olahraga
• Monitor gula darah mingguan
• Pertimbangkan konsultasi nutrisionis"""
        elif risk_prob >= 25:
            interpretasi = """RISIKO SEDANG:
• Pemeriksaan kesehatan rutin setiap 3-6 bulan
• Perhatikan pola makan dan aktivitas fisik
• Monitor berat badan dan BMI
• Tes gula darah berkala
• Edukasi pencegahan diabetes"""
        else:
            interpretasi = """RISIKO RENDAH:
• Pertahankan gaya hidup sehat
• Pemeriksaan tahunan tetap dianjurkan
• Jaga berat badan ideal
• Olahraga teratur minimal 150 menit/minggu
• Diet seimbang dan bergizi"""
            
        self.result_display.insert(tk.END, interpretasi + "\n\n")
        
        self.result_display.insert(tk.END, "-"*60 + "\n")
        self.result_display.insert(tk.END, "FAKTOR RISIKO YANG PERLU DIPERHATIKAN:\n")
        self.result_display.insert(tk.END, "-"*60 + "\n")
        
        if 'BMI' in input_data and input_data['BMI'] > 25:
            self.result_display.insert(tk.END, "• BMI tinggi - pertimbangkan penurunan berat badan\n")
        if 'Glucose' in input_data and input_data['Glucose'] > 100:
            self.result_display.insert(tk.END, "• Glukosa tinggi - perlu monitoring ketat\n")
        if 'Age' in input_data and input_data['Age'] > 45:
            self.result_display.insert(tk.END, "• Usia >45 tahun - risiko meningkat\n")
        if 'BloodPressure' in input_data and input_data['BloodPressure'] > 80:
            self.result_display.insert(tk.END, "• Tekanan darah perlu diperhatikan\n")
            
        self.result_display.insert(tk.END, "\n" + "="*60 + "\n")
        self.result_display.insert(tk.END, "CATATAN PENTING:\n")
        self.result_display.insert(tk.END, "Hasil prediksi ini berdasarkan model machine learning\n")
        self.result_display.insert(tk.END, "dan bukan diagnosis medis definitif. Konsultasikan\n")
        self.result_display.insert(tk.END, "dengan tenaga medis untuk diagnosis yang akurat.\n")
        self.result_display.insert(tk.END, "="*60 + "\n")
        
    def show_welcome_message(self):
        self.result_display.insert(tk.END, "="*60 + "\n")
        self.result_display.insert(tk.END, "    SELAMAT DATANG DI SISTEM PREDIKSI DIABETES\n")
        self.result_display.insert(tk.END, "="*60 + "\n\n")
        
        self.result_display.insert(tk.END, "Sistem ini menggunakan algoritma K-Nearest Neighbors\n")
        self.result_display.insert(tk.END, "untuk memprediksi risiko diabetes berdasarkan data klinis.\n\n")
        
        self.result_display.insert(tk.END, "CARA PENGGUNAAN:\n")
        self.result_display.insert(tk.END, "-"*40 + "\n")
        self.result_display.insert(tk.END, "1. Klik 'Pilih Dataset' untuk memuat data CSV\n")
        self.result_display.insert(tk.END, "2. Klik 'Training Model' untuk melatih model\n")
        self.result_display.insert(tk.END, "3. Isi form dengan data pasien\n")
        self.result_display.insert(tk.END, "4. Klik 'PREDIKSI RISIKO' untuk hasil\n")
        self.result_display.insert(tk.END, "5. Gunakan 'RESET FORM' untuk data baru\n\n")
        
        self.result_display.insert(tk.END, "="*60 + "\n")
        
    def reset_form(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        
        self.show_welcome_message()
        
def main():
    root = tk.Tk()
    app = SistemPrediksiDiabetes(root)
    root.mainloop()

if __name__ == "__main__":
    main()