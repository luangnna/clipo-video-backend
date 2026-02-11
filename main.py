
import subprocess

def cortar_video(input_path, output_path, start, duration):
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-ss", str(start),
        "-t", str(duration),
        "-c", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Vídeo cortado com sucesso: {output_path}")

# Exemplo de uso
cortar_video("videos/meu_video.mp4", "videos/corte.mp4", 10, 15)
