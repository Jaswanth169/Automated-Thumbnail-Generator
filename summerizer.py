import cv2
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFont, ImageDraw
import azure.cognitiveservices.speech as speechsdk
from moviepy.editor import VideoFileClip
import time
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

speech_key = os.getenv('SPEECH_KEY')
speech_region = os.getenv('SPEECH_REGION')
nvapi_key = os.getenv('NVAPI_KEY')  # Add your NVIDIA API key to your .env file

output_file = "transcription.txt"  # Define your output file name here

# Function to extract frames from a video
def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return 0
    os.makedirs(output_folder, exist_ok=True)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"frame_{frame_count}.jpg"
        frame_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    print(f"{frame_count} frames extracted to {output_folder}")
    return frame_count

# Function to calculate information/content metric of a frame
def frame_information(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    return var

# Function to adjust contrast of an image using PIL
def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

# Function to find and save frame with the most information
def extract_best_frame(frames_folder, output_folder):
    max_info_frame = None
    max_info_value = -1
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(frames_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            frame_path = os.path.join(frames_folder, filename)
            frame = cv2.imread(frame_path)
            info_value = frame_information(frame)
            if info_value > max_info_value:
                max_info_value = info_value
                max_info_frame = frame
    if max_info_frame is not None:
        pil_image = Image.fromarray(cv2.cvtColor(max_info_frame, cv2.COLOR_BGR2RGB))
        enhanced_pil_image = adjust_contrast(pil_image, 1.5)
        enhanced_frame = cv2.cvtColor(np.array(enhanced_pil_image), cv2.COLOR_RGB2BGR)
        output_path = os.path.join(output_folder, 'best_frame_enhanced.jpg')
        cv2.imwrite(output_path, enhanced_frame)
        print(f"Best frame with enhanced contrast saved to {output_path}")
        return output_path
    else:
        print("No frames found in the input folder")
        return None

# Function to extract transcription from video using Azure Cognitive Services
def video_to_wav(video_file):
    try:
        output_wav = os.path.splitext(video_file)[0] + '.wav'
        video = VideoFileClip(video_file)
        audio = video.audio
        audio.write_audiofile(output_wav)
        audio.close()
        video.close()
        return output_wav
    except Exception as e:
        print(f"Error converting video to WAV: {str(e)}")
        return None

def conversation_transcriber_transcribed_cb(evt):
    try:
        with open(output_file, 'a', encoding='utf-8') as file:
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                file.write(f"{evt.result.text}\n")
    except Exception as e:
        print(f"Error in transcription callback: {e}")

def recognize_from_file(video_path):
    try:
        # Convert video to WAV format
        wav_file = video_to_wav(video_path)
        if not wav_file:
            print("Failed to convert video to WAV.")
            return
        
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        speech_config.speech_recognition_language = "en-US"

        audio_config = speechsdk.audio.AudioConfig(filename=wav_file)
        conversation_transcriber = speechsdk.transcription.ConversationTranscriber(speech_config=speech_config, audio_config=audio_config)

        transcribing_stop = False

        def stop_cb(evt):
            print('CLOSING on {}'.format(evt))
            nonlocal transcribing_stop
            transcribing_stop = True

        # Connect callbacks to the events fired by the conversation transcriber
        conversation_transcriber.transcribed.connect(conversation_transcriber_transcribed_cb)
        conversation_transcriber.session_started.connect(lambda evt: print('SessionStarted event'))
        conversation_transcriber.session_stopped.connect(lambda evt: print('SessionStopped event'))
        conversation_transcriber.canceled.connect(lambda evt: print('Canceled event'))
        conversation_transcriber.session_stopped.connect(stop_cb)
        conversation_transcriber.canceled.connect(stop_cb)

        conversation_transcriber.start_transcribing_async()

        # Waits for completion.
        while not transcribing_stop:
            time.sleep(.5)

        conversation_transcriber.stop_transcribing_async()

        print(f"Transcription saved to {output_file}")

    except Exception as err:
        print(f"Encountered exception: {err}")

# Function to summarize transcription and generate title using NVIDIA API
def generate_title(transcription):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {nvapi_key}'
    }

    data = {
        "model": "mistralai/mistral-7b-instruct-v0.3",
        "messages": [{"role": "user", "content": f"Generate a YouTube video title based on the following transcription: {transcription}"}],
        "temperature": 0.7,
        "top_p": 0.7,
        "max_tokens": 60
    }

    response = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return "Title Generation Failed"

# Function to get contrasting color for text
def get_contrasting_color(color):
    luminance = (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]) / 255
    return (0, 0, 0) if luminance > 0.5 else (255, 255, 255)

# Function to find maximum font size for text
def find_max_font_size(draw, text, image_width, font_path, max_font_size):
    for font_size in range(1, max_font_size):
        font = ImageFont.truetype(font_path, font_size)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        if text_width >= image_width:
            return font_size - 1
    return max_font_size

# Function to overlay text on image
def overlay_text_on_image(image_path, text, font_path, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    max_font_size = find_max_font_size(draw, text, image.width, font_path, 200)
    font = ImageFont.truetype(font_path, max_font_size)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (image.width - text_width) // 2
    text_y = (image.height - text_height) // 2
    background_color = image.getpixel((text_x + text_width // 2, text_y + text_height // 2))
    text_color = get_contrasting_color(background_color)
    draw.text((text_x, text_y), text, text_color, font=font)
    image.save(output_path)

# Main function to orchestrate the thumbnail generation process
def main():
    video_path = input("Enter the path of the video file: ")

    frames_folder = 'frames_output'
    output_folder = 'output_frame'
    font_path = 'Paintingwithchocolate-K5mo.ttf'

    while True:
        # Step 1: Extract frames from the video
        frame_count = extract_frames(video_path, frames_folder)

        if frame_count == 0:
            print("No frames extracted. Exiting.")
            return

        # Step 2: Extract the best frame
        best_frame_path = extract_best_frame(frames_folder, output_folder)

        if best_frame_path is None:
            print("No informative frames found. Exiting.")
            return

        # Step 3: Extract transcription from the video using Azure Cognitive Services
        recognize_from_file(video_path)

        # Step 4: Read the transcription from the file
        with open(output_file, 'r', encoding='utf-8') as f:
            transcription = f.read()

        # Step 5: Prompt user to choose how to generate title
        while True:
            title_choice = input("Do you want to generate the title automatically using NVIDIA LLM? (Y/N): ").strip().lower()
            if title_choice == 'y':
                title = generate_title(transcription)
                break
            elif title_choice == 'n':
                title = input("Enter your custom title for the thumbnail: ")
                break
            else:
                print("Invalid choice. Please enter 'Y' or 'N'.")

        # Step 6: Overlay title on the best frame to create the thumbnail
        thumbnail_path = 'generated_thumbnail.jpg'
        overlay_text_on_image(best_frame_path, title, font_path, thumbnail_path)

        # Step 7: Display the generated thumbnail
        print(f"Thumbnail generated: {thumbnail_path}")

        # Step 8: Ask for user feedback on the thumbnail
        feedback = input("Do you like the generated thumbnail? (Y/N): ").strip().lower()
        if feedback != 'y':
            print("Generating a new thumbnail...")
            continue
        else:
            print("Thumbnail process completed.")
            break

if __name__ == "__main__":
    main()
