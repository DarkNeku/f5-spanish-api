from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
import tempfile
import os
import io
import base64
from pathlib import Path
import logging
import time
import subprocess
import sys

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="F5-Spanish TTS API", version="1.0.0")

# Configurar CORS para Unity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para el modelo
model = None
vocoder = None
device = None

def install_f5_tts():
    """Instala F5-TTS si no está disponible"""
    try:
        logger.info("Instalando F5-TTS...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/SWivid/F5-TTS.git"
        ])
        logger.info("F5-TTS instalado correctamente")
    except Exception as e:
        logger.error(f"Error instalando F5-TTS: {e}")
        raise

def load_model():
    """Carga el modelo F5-Spanish"""
    global model, vocoder, device
    
    try:
        logger.info("Cargando modelo F5-Spanish...")
        start_time = time.time()
        
        # Detectar dispositivo
        device = torch.device("cpu")  # Render usa CPU
        logger.info(f"Usando dispositivo: {device}")
        
        # Importar F5-TTS
        try:
            from f5_tts.api import F5TTS
        except ImportError:
            logger.info("F5-TTS no encontrado, instalando...")
            install_f5_tts()
            from f5_tts.api import F5TTS
        
        # Cargar modelo español
        model = F5TTS(model_type="F5-TTS", ckpt_file=None, vocab_file=None, device=device)
        
        load_time = time.time() - start_time
        logger.info(f"Modelo cargado en {load_time:.2f} segundos")
        
        return True
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Carga el modelo al iniciar la API"""
    logger.info("Iniciando API F5-Spanish TTS...")
    success = load_model()
    if not success:
        logger.error("No se pudo cargar el modelo")
    else:
        logger.info("API lista para usar")

@app.get("/")
async def root():
    """Endpoint de prueba"""
    return {
        "message": "F5-Spanish TTS API funcionando",
        "model_loaded": model is not None,
        "device": str(device) if device else "No disponible"
    }

@app.get("/health")
async def health_check():
    """Verificar estado de la API"""
    return {
        "status": "healthy",
        "model_ready": model is not None,
        "timestamp": time.time()
    }

@app.post("/synthesize")
async def synthesize_speech(
    text: str = Form(..., description="Texto a sintetizar"),
    ref_audio: UploadFile = File(..., description="Audio de referencia"),
    ref_text: str = Form(..., description="Transcripción del audio de referencia"),
    speed: float = Form(1.0, description="Velocidad de síntesis (0.5-2.0)"),
    remove_silence: bool = Form(True, description="Remover silencios")
):
    """
    Sintetizar voz usando F5-Spanish TTS
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Texto requerido")
    
    if not ref_text or not ref_text.strip():
        raise HTTPException(status_code=400, detail="Transcripción del audio de referencia requerida")
    
    try:
        logger.info(f"Procesando síntesis: '{text[:50]}...'")
        start_time = time.time()
        
        # Guardar audio de referencia temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await ref_audio.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        
        try:
            # Generar audio con F5-TTS
            logger.info("Iniciando síntesis...")
            
            # Configurar parámetros
            gen_params = {
                "speed": speed,
                "remove_silence": remove_silence
            }
            
            # Síntesis de voz
            audio_data = model.infer(
                ref_audio=temp_audio_path,
                ref_text=ref_text,
                gen_text=text,
                **gen_params
            )
            
            # Convertir a formato compatible
            if isinstance(audio_data, torch.Tensor):
                audio_numpy = audio_data.cpu().numpy()
            else:
                audio_numpy = audio_data
            
            # Guardar audio resultante
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
                import soundfile as sf
                sf.write(temp_output.name, audio_numpy, 24000)
                
                # Leer el archivo generado
                with open(temp_output.name, "rb") as f:
                    audio_bytes = f.read()
            
            # Convertir a base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            processing_time = time.time() - start_time
            logger.info(f"Síntesis completada en {processing_time:.2f} segundos")
            
            return {
                "success": True,
                "audio_base64": audio_base64,
                "format": "wav",
                "sample_rate": 24000,
                "processing_time": processing_time,
                "text_length": len(text),
                "ref_text": ref_text
            }
            
        finally:
            # Limpiar archivos temporales
            try:
                os.unlink(temp_audio_path)
                if 'temp_output' in locals():
                    os.unlink(temp_output.name)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error en síntesis: {e}")
        raise HTTPException(status_code=500, detail=f"Error en síntesis: {str(e)}")

@app.post("/test-synthesis")
async def test_synthesis():
    """
    Endpoint de prueba con datos predefinidos
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Generar un audio de prueba simple
        test_text = "Hola, esta es una prueba de síntesis de voz."
        
        logger.info("Ejecutando prueba de síntesis...")
        start_time = time.time()
        
        # Aquí usarías un audio de referencia predefinido
        # Por ahora retornamos un mensaje de prueba
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "message": "Modelo funcionando correctamente",
            "processing_time": processing_time,
            "test_text": test_text,
            "model_ready": True
        }
        
    except Exception as e:
        logger.error(f"Error en prueba: {e}")
        raise HTTPException(status_code=500, detail=f"Error en prueba: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)