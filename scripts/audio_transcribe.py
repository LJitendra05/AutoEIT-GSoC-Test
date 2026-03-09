import whisper
import pandas as pd
import os
import re
#load data
data_folder=r'data\audiodata'
all_dfs=[]
for file in  os.listdir(data_folder):
    if file.endswith('.csv'):
        path=os.path.join(data_folder,file)
        df=pd.read_csv(path,encoding='utf-8')
        df=df.iloc[0:30,:3]
        df['participant']=file #mark which participant(file)
        all_dfs.append(df)
data=pd.concat(all_dfs,ignore_index=True)
data.head()

#clear
def clean(text):
    return re.sub(r"\s*\(.*?\)","",str(text)).strip()

data['Stimulus']=data['Stimulus'].apply(clean)

#load model
model=whisper.load_model('small')

folder = r'data\audio'
transcriptions = []
for file in os.listdir(folder): 
    if file.endswith(".mp3"):    
        audio_path = os.path.join(folder, file)  
        print("processing",file)
        result = model.transcribe(audio_path, language="es")
        full_text = result["text"]
        sentences = re.split(r'[.!?]', full_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        transcriptions.extend(sentences)

data["Transcription"].iloc[:len(transcriptions)] = transcriptions  #as the len(transcription) <120

# Save results
output_path = "results/participant_transcription.csv"
data.to_csv(output_path, index=False)

print("transcribe complete. Results saved to:", output_path)
