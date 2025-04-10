import streamlit as st
from modules.pipelines.speaker_role_inference import SpeakerRoleInferencePipeline
import os

# Create file upload widget
st.title("Speech Processing and Role Inference")
st.subheader("Upload your audio file (e.g., .wav, .mp3, .m4a)")

audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    # Extract the file name without extension and append .wav
    file_name = os.path.splitext(audio_file.name)[0] + ".wav"
    
    # Save the uploaded file locally with the new name
    audio_file_path = os.path.join("uploads", file_name)
    
    # Ensure the uploads directory exists
    os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)
    
    with open(audio_file_path, "wb") as f:
        f.write(audio_file.getbuffer())
    
    # Show progress bar for file processing
    progress = st.progress(0)  # Initialize progress bar at 0%
    
    # Step 1: Run initial setup or preprocessing
    st.write("Starting audio processing...")
    progress.progress(10)  # Update progress to 10%

    # Step 2: Run the pipeline
    pipeline = SpeakerRoleInferencePipeline(audio_file_path=audio_file_path)  # Pass the audio file path here
    
    st.write("Running speaker role inference pipeline...")
    progress.progress(50)  # Update progress to 50%

    role_mapping = pipeline.run()
    
    # Step 3: Display results
    progress.progress(80)  # Update progress to 80%
    
    st.subheader("Role Mapping")
    st.json(role_mapping)
    
    # Final step: Notify the user when processing is complete
    progress.progress(100)  # Update progress to 100%
    st.success("Pipeline completed successfully!")
