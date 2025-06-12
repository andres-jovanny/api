# api
api de microexpresiones
# -*- coding: utf-8 -*-
"""
Detector de Emociones Faciales en Tiempo Real
Versión adaptada para ejecución local
"""

import cv2
import numpy as np
from fer import FER
import time

class EmotionDetector:
    def __init__(self):
        # Inicializar el detector con MTCNN para mayor precisión
        self.detector = FER(mtcnn=True)
        
        # Configuración de la cámara
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        self.target_fps = 15
        
        # Estadísticas
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0
        self.last_emotion = None
        self.running = False
        
        # Configuración de visualización
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        self.rectangle_color = (0, 255, 0)  # Verde
        self.text_color = (0, 0, 255)       # Rojo
        
    def initialize_camera(self):
        """Inicializa la cámara web"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo acceder a la cámara")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
    def process_frame(self, frame):
        """Procesa un frame para detectar emociones"""
        try:
            # Convertir a RGB para el detector
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detectar emociones
            faces = self.detector.detect_emotions(rgb_img)
            current_emotion = None
            
            # Dibujar resultados en la imagen
            for face in faces:
                x, y, w, h = face['box']
                
                # Asegurar que las coordenadas estén dentro de los límites
                x, y = max(0, x), max(0, y)
                w, h = min(w, self.frame_width - x), min(h, self.frame_height - y)
                
                # Dibujar rectángulo alrededor del rostro
                cv2.rectangle(frame, (x, y), (x+w, y+h), self.rectangle_color, 2)
                
                # Obtener emoción dominante
                emotion, confidence = max(face['emotions'].items(), key=lambda x: x[1])
                current_emotion = (emotion, confidence)
                self.last_emotion = current_emotion
                
                # Mostrar texto con la emoción y confianza
                text = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y-10), 
                           self.font, self.font_scale, 
                           (0, 0, 0), self.font_thickness + 2)  # Borde negro
                cv2.putText(frame, text, (x, y-10), 
                           self.font, self.font_scale, 
                           self.text_color, self.font_thickness)  # Texto rojo
            
            return frame, current_emotion
            
        except Exception as e:
            print(f"Error al procesar frame: {str(e)}")
            return frame, None
    
    def calculate_fps(self):
        """Calcula los FPS actuales"""
        elapsed_time = time.time() - self.start_time
        self.fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        return self.fps
    
    def display_info(self, frame):
        """Muestra información en el frame"""
        # Mostrar FPS
        fps_text = f"FPS: {self.calculate_fps():.1f}"
        cv2.putText(frame, fps_text, (10, 30), 
                   self.font, self.font_scale, 
                   self.text_color, self.font_thickness)
        
        # Mostrar última emoción detectada
        if self.last_emotion:
            emotion_text = f"Emoción: {self.last_emotion[0]} ({self.last_emotion[1]:.2f})"
            cv2.putText(frame, emotion_text, (10, 60), 
                       self.font, self.font_scale, 
                       self.text_color, self.font_thickness)
        
        # Mostrar instrucciones
        instructions = "Presiona 'Q' para salir"
        cv2.putText(frame, instructions, (10, self.frame_height - 20), 
                   self.font, 0.5, 
                   (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Bucle principal de detección"""
        try:
            self.initialize_camera()
            self.running = True
            self.start_time = time.time()
            self.frame_count = 0
            
            print("Iniciando detector de emociones...")
            print("Presiona 'Q' para salir")
            
            while self.running:
                # Capturar frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error al capturar frame")
                    break
                
                # Procesar frame
                processed_frame, _ = self.process_frame(frame)
                
                # Mostrar información
                processed_frame = self.display_info(processed_frame)
                
                # Mostrar frame
                cv2.imshow('Detector de Emociones', processed_frame)
                
                # Incrementar contador de frames    
                self.frame_count += 1
                
                # Controlar velocidad de procesamiento
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                
        except Exception as e:
            print(f"Error durante la ejecución: {str(e)}")
        finally:
            # Liberar recursos
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Detector detenido")

if __name__ == "__main__":
    # Instrucciones de instalación de dependencias
    print("""
    Antes de ejecutar, asegúrate de tener instaladas las dependencias:
    pip install numpy opencv-python fer matplotlib tensorflow keras
    """)
    
    # Crear y ejecutar detector
    detector = EmotionDetector()
    detector.run()
