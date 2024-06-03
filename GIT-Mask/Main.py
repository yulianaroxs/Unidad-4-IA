from ExtraccionFrames import process_all_videos
from PromediarFrames import process_all_averaging
from ConversionCSV import process_all_to_csv
from EntrenarCNN import train_model
from predict_person import predict_person  # Asegúrate de que esta función esté disponible

def main():
    # Extraer frames de los videos
    process_all_videos('/content/Proyecto-IA-U4/GIT-Mask/video-outputs', 'GIT-Mask/GIT-Mask/frames', 5)
    
    # Promediar los frames extraídos
    process_all_averaging('/content/Proyecto-IA-U4/GIT-Mask/frames', 'GIT-Mask/GIT-Mask/averaged')
    
    # Convertir los frames promediados a archivos CSV
    process_all_to_csv('/content/Proyecto-IA-U4/GIT-Mask/averaged', 'GIT-Mask/GIT-Mask/csv')
    
    # Entrenar la red CNN con los datos CSV
    train_model('/content/Proyecto-IA-U4/GIT-Mask/csv')
    
    # Predecir la persona en un nuevo video
    predicted_person = predict_person(
        video_path='/ruta/a/tu/nuevo/video.avi',
        model_path='trained_model.h5',
        frames_per_second=5,
        img_height=480,
        img_width=848,
        channels=3,
        window_size=5
    )
    print(f'Predicted person: {predicted_person}')
    
    print("Proceso completo: Extracción, Promedio, Conversión, Entrenamiento y Predicción de la CNN.")

if __name__ == '__main__':
    main()
