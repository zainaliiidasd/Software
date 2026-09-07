import os
import sys
import json
import threading
import requests
from pathlib import Path
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

# Core AI & Media Libraries
import whisper
import numpy as np
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from moviepy.video.fx.all import fadein, fadeout
import google.generativeai as genai

# Theme Configuration
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ==============================================================================
# 1. DIRECTORY & MODEL MANAGEMENT SYSTEM
# ==============================================================================
BASE_DIR = Path.home() / ".autocut_ai_suite"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "whisper_large_v3": {
        "name": "Whisper Large v3 (Transcription)",
        "type": "whisper",
        "size_mb": 2900,
        "filename": "large-v3.pt"
    },
    "f5_tts": {
        "name": "F5-TTS (Voice Cloning)",
        "type": "tts",
        "size_mb": 1800,
        "filename": "f5_tts_model.pt"
    }
}

def is_model_installed(model_key):
    config = MODEL_CONFIGS.get(model_key)
    if not config:
        return False
    return (MODELS_DIR / config["filename"]).exists()

def delete_local_model(model_key):
    config = MODEL_CONFIGS.get(model_key)
    if config:
        target_path = MODELS_DIR / config["filename"]
        if target_path.exists():
            os.remove(target_path)
            return True
    return False

# ==============================================================================
# 2. BACKEND PROCESSING ENGINE
# ==============================================================================
class BackendEngine:
    @staticmethod
    def transcribe_audio(audio_path, progress_callback=None):
        if not is_model_installed("whisper_large_v3"):
            if progress_callback:
                progress_callback("Downloading Whisper Model...")
        
        model = whisper.load_model("large-v3", download_root=str(MODELS_DIR))
        result = model.transcribe(audio_path, word_timestamps=True)
        return result

    @staticmethod
    def generate_viral_script(api_key, topic):
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Act as an expert viral video creator. Write a high-retention script on: '{topic}'. Include scene descriptions."
        response = model.generate_content(prompt)
        return response.text

    @staticmethod
    def apply_ken_burns_zoom(image_clip, zoom_ratio=0.08):
        d = image_clip.duration
        return image_clip.fl(lambda gf, t: ImageClip(gf(t)).resize(1 + zoom_ratio * (t / d)).get_frame(t))

    def assemble_vip_video(self, image_paths, voiceover_path, output_path, apply_zoom=True, apply_crossfade=True, status_callback=None):
        if status_callback:
            status_callback("Loading audio & calculating timing...")
            
        audio = AudioFileClip(voiceover_path)
        total_duration = audio.duration
        num_images = len(image_paths)
        if num_images == 0:
            raise ValueError("No images selected.")

        per_image_duration = total_duration / num_images
        clips = []

        for idx, img_path in enumerate(image_paths):
            if status_callback:
                status_callback(f"Processing clip {idx + 1}/{num_images}...")
                
            clip = ImageClip(img_path).set_duration(per_image_duration)
            
            if apply_zoom:
                clip = self.apply_ken_burns_zoom(clip, zoom_ratio=0.08)
                
            if apply_crossfade:
                if idx > 0:
                    clip = clip.fx(fadein, duration=0.4)
                clip = clip.fx(fadeout, duration=0.4)
                
            clips.append(clip)

        if status_callback:
            status_callback("Rendering final 1080p video...")

        final_video = concatenate_videoclips(clips, method="compose")
        final_video = final_video.set_audio(audio)
        final_video.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac", preset="medium")
        
        if status_callback:
            status_callback("Render Complete!")
        return output_path

# ==============================================================================
# 3. UNIFIED CUSTOMTKINTER GUI
# ==============================================================================
class AutoCutApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AutoCut AI Suite - VIP Content Platform")
        self.geometry("1100x700")
        self.minsize(950, 600)

        self.engine = BackendEngine()
        self.selected_audio = ""
        self.selected_images = []

        # Grid Setup
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------------- SIDEBAR ----------------
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.logo_label = ctk.CTkLabel(self.sidebar, text="AUTOCUT AI", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.pack(padx=20, pady=(25, 30))

        # Navigation Buttons
        self.btn_studio = self._create_nav_btn("🎬 AutoCut Studio PRO", lambda: self.select_tab("studio"))
        self.btn_transcribe = self._create_nav_btn("🎙️ AI Transcriber", lambda: self.select_tab("transcribe"))
        self.btn_script = self._create_nav_btn("✍️ AI Scriptwriter", lambda: self.select_tab("script"))
        self.btn_clone = self._create_nav_btn("🗣️ Voice Cloning", lambda: self.select_tab("clone"))
        self.btn_models = self._create_nav_btn("⚙️ Model Manager", lambda: self.select_tab("models"))

        # Theme Switch
        self.theme_switch = ctk.CTkSwitch(self.sidebar, text="Dark Mode", command=self.toggle_theme)
        self.theme_switch.pack(side="bottom", padx=20, pady=20)
        self.theme_switch.select()

        # ---------------- MAIN VIEW ----------------
        self.content_frame = ctk.CTkFrame(self, corner_radius=12, fg_color="transparent")
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        self.select_tab("studio")

    def _create_nav_btn(self, text, command):
        btn = ctk.CTkButton(self.sidebar, text=text, anchor="w", fg_color="transparent", 
                            text_color=("gray10", "gray90"), height=35, command=command)
        btn.pack(fill="x", padx=10, pady=4)
        return btn

    def toggle_theme(self):
        ctk.set_appearance_mode("Dark" if self.theme_switch.get() == 1 else "Light")

    def clear_content(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def select_tab(self, tab_name):
        self.clear_content()
        if tab_name == "studio":
            self.build_studio_tab()
        elif tab_name == "models":
            self.build_models_tab()
        elif tab_name == "script":
            self.build_script_tab()
        else:
            label = ctk.CTkLabel(self.content_frame, text=f"{tab_name.title()} Module Interface Ready", font=ctk.CTkFont(size=18))
            label.pack(pady=50)

    # ------------ AUTOCUT STUDIO PRO TAB ------------
    def build_studio_tab(self):
        ctk.CTkLabel(self.content_frame, text="AutoCut Studio PRO (VIP Sync & FX)", font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w", pady=(0, 15))

        # Card 1: File Selection
        card1 = ctk.CTkFrame(self.content_frame, corner_radius=10)
        card1.pack(fill="x", pady=8, ipady=5)

        self.lbl_audio = ctk.CTkLabel(card1, text="No Audio Selected", text_color="gray")
        self.lbl_audio.pack(side="left", padx=15)
        ctk.CTkButton(card1, text="Select Voiceover", command=self.choose_audio).pack(side="right", padx=15, pady=10)

        card2 = ctk.CTkFrame(self.content_frame, corner_radius=10)
        card2.pack(fill="x", pady=8, ipady=5)

        self.lbl_images = ctk.CTkLabel(card2, text="No Images Selected", text_color="gray")
        self.lbl_images.pack(side="left", padx=15)
        ctk.CTkButton(card2, text="Select Images", command=self.choose_images).pack(side="right", padx=15, pady=10)

        # Card 2: Options
        card_opts = ctk.CTkFrame(self.content_frame, corner_radius=10)
        card_opts.pack(fill="x", pady=8, ipady=10)

        self.var_zoom = ctk.BooleanVar(value=True)
        self.var_crossfade = ctk.BooleanVar(value=True)

        ctk.CTkCheckBox(card_opts, text="Auto-Ken Burns Zoom Animation", variable=self.var_zoom).pack(anchor="w", padx=15, pady=5)
        ctk.CTkCheckBox(card_opts, text="Smooth Crossfade Transitions", variable=self.var_crossfade).pack(anchor="w", padx=15, pady=5)

        # Render Controls
        self.btn_render = ctk.CTkButton(self.content_frame, text="🚀 Render VIP Video (1080p)", font=ctk.CTkFont(size=15, weight="bold"), height=42, command=self.start_render_thread)
        self.btn_render.pack(fill="x", pady=(20, 10))

        self.lbl_status = ctk.CTkLabel(self.content_frame, text="Ready", text_color="gray")
        self.lbl_status.pack(anchor="w")

        self.progress_bar = ctk.CTkProgressBar(self.content_frame)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", pady=5)

    def choose_audio(self):
        path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.m4a")])
        if path:
            self.selected_audio = path
            self.lbl_audio.configure(text=Path(path).name, text_color="white")

    def choose_images(self):
        paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if paths:
            self.selected_images = list(paths)
            self.lbl_images.configure(text=f"{len(paths)} Images Selected", text_color="white")

    def start_render_thread(self):
        if not self.selected_audio or not self.selected_images:
            messagebox.showerror("Error", "Please select both a voiceover file and images.")
            return

        output_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
        if not output_path:
            return

        self.btn_render.configure(state="disabled")
        self.progress_bar.set(0.2)
        
        def run():
            try:
                self.engine.assemble_vip_video(
                    self.selected_images, 
                    self.selected_audio, 
                    output_path, 
                    apply_zoom=self.var_zoom.get(), 
                    apply_crossfade=self.var_crossfade.get(),
                    status_callback=lambda msg: self.lbl_status.configure(text=msg)
                )
                self.progress_bar.set(1.0)
                messagebox.showinfo("Success", "Video successfully rendered!")
            except Exception as e:
                messagebox.showerror("Render Error", str(e))
            finally:
                self.btn_render.configure(state="normal")

        threading.Thread(target=run, daemon=True).start()

    # ------------ SCRIPTWRITER TAB ------------
    def build_script_tab(self):
        ctk.CTkLabel(self.content_frame, text="Viral AI Scriptwriter", font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w", pady=(0, 15))

        ctk.CTkLabel(self.content_frame, text="Gemini API Key:").pack(anchor="w")
        entry_key = ctk.CTkEntry(self.content_frame, show="*")
        entry_key.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(self.content_frame, text="Video Topic:").pack(anchor="w")
        entry_topic = ctk.CTkEntry(self.content_frame, placeholder_text="e.g. 5 Mind-Blowing AI Tools in 2026")
        entry_topic.pack(fill="x", pady=(0, 10))

        txt_output = ctk.CTkTextbox(self.content_frame, height=250)
        
        def generate():
            key = entry_key.get().strip()
            topic = entry_topic.get().strip()
            if not key or not topic:
                messagebox.showwarning("Warning", "API Key and Topic are required.")
                return
            try:
                res = self.engine.generate_viral_script(key, topic)
                txt_output.delete("1.0", "end")
                txt_output.insert("1.0", res)
            except Exception as e:
                messagebox.showerror("Error", str(e))

        ctk.CTkButton(self.content_frame, text="Generate Script", command=generate).pack(fill="x", pady=10)
        txt_output.pack(fill="both", expand=True)

    # ------------ MODEL MANAGER TAB ------------
    def build_models_tab(self):
        ctk.CTkLabel(self.content_frame, text="Local AI Models Manager", font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w", pady=(0, 15))

        for key, info in MODEL_CONFIGS.items():
            card = ctk.CTkFrame(self.content_frame, corner_radius=10)
            card.pack(fill="x", pady=6, padx=5)

            status = "Installed" if is_model_installed(key) else "Not Installed"
            color = "green" if is_model_installed(key) else "gray"

            ctk.CTkLabel(card, text=f"{info['name']} [{info['size_mb']} MB]", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=15, pady=15)
            ctk.CTkLabel(card, text=status, text_color=color).pack(side="left", padx=10)

            def make_delete_cmd(k=key):
                def cmd():
                    if delete_local_model(k):
                        messagebox.showinfo("Success", "Model deleted successfully.")
                        self.build_models_tab()
                    else:
                        messagebox.showwarning("Warning", "Model file not found.")
                return cmd

            ctk.CTkButton(card, text="Delete", fg_color="#E74C3C", hover_color="#C0392B", width=90, command=make_delete_cmd(key)).pack(side="right", padx=15)

if __name__ == "__main__":
    app = AutoCutApp()
    app.mainloop()
