# end? 2.0
!pip install ffmpeg-python
!pip -q install librosa soundfile plotly

from IPython.display import HTML, Audio, display
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import soundfile as sf



AUDIO_HTML = """
<script>
window.recordAudio = function() {
  return new Promise((resolve, reject) => {
    var btn = document.createElement("BUTTON");
    btn.innerText = "Start Recording";
    document.body.appendChild(btn);
    
    btn.onclick = function() {
      if (btn.innerText === "Start Recording") {
        btn.innerText = "Stop Recording";
        navigator.mediaDevices.getUserMedia({audio: true}).then(stream => {
          var recorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
          
          recorder.ondataavailable = function(e) {
            var reader = new FileReader();
            reader.readAsDataURL(e.data);
            reader.onloadend = function() {
              var base64data = reader.result;
              resolve(base64data);
              btn.remove();
            }
          };
          
          recorder.start();
          
          btn.onclick = function() {
            recorder.stop();
            stream.getAudioTracks()[0].stop();
          };
        }).catch(err => reject(err));
      }
    };
  });
};
</script>
"""

'''def get_audio():
    display(HTML(AUDIO_HTML))
    data = eval_js("recordAudio()")
    binary = b64decode(data.split(',')[1])
    
    try:
        # Конвертируем WebM в WAV с помощью ffmpeg
        output, err = ffmpeg.input('pipe:', format='webm') \
                         .output('pipe:', format='wav') \
                         .run(capture_stdout=True, capture_stderr=True, input=binary)
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise
    
    # Читаем WAV-файл
    sr, audio = wav_read(io.BytesIO(output))
    return audio, sr

# Пример использования
audio, sr = get_audio()
display(Audio(audio, rate=sr))

# Создаем график
t_frame = len(audio) / sr
t = np.linspace(0, t_frame, len(audio), endpoint=False)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=t,
    y=audio.flatten(),  # Flatten if multi-channel
    mode='lines',
    name='Audio Waveform',
    line=dict(color='red')
))

fig.update_layout(
    template='plotly_white',
    width=800,
    height=600,
    title='Audio Waveform',
    xaxis_title='Time (s)',
    yaxis_title='Amplitude'
)

fig.show()'''

# работа со звуком

def avg_squared_deviation(x, axis=None, ddof=0, ignore_nan=True, keepdims=False): # вычислили среднее кв отклонениеб ddof  это значит. что ссчитаем по все данным 

    x = np.asarray(x, dtype=float)
    if ignore_nan:
        return np.nanvar(x, axis=axis, ddof=ddof, keepdims=keepdims)
    else:
        return np.var(x, axis=axis, ddof=ddof, keepdims=keepdims)

def rms_deviation(x, axis=None, ddof=0, ignore_nan=True, keepdims=False): # nan - значит что случается ошибка в вычислении и программа ее игнорирует
    """
    Стандартное отклонение (RMS) = sqrt(variance)
    """
    var = avg_squared_deviation(x, axis=axis, ddof=ddof, ignore_nan=ignore_nan, keepdims=keepdims)
    return np.sqrt(var)

def equalize_amplitudes_Zxx(Zxx, method='per_freq', mode='outlier', blend=1.0, ddof=0, ignore_nan=True):  # stif как частотный состав сигнала меняется во времениб с преобразованиями фурье
 
    S = np.abs(Zxx)  # Амплитуда
    phase = np.angle(Zxx)  # Фаза
