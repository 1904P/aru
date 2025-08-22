# end? 2.0
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
import sounddevice as sd
from scipy.io.wavfile import write, read
import matplotlib.pyplot as plt

def get_audio(duration=5, fs=44100):
    print(f"Recording {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished")

    # Сохраняем во временный WAV в памяти
    buf = io.BytesIO()
    write(buf, fs, audio)
    buf.seek(0)

    # Читаем WAV из памяти
    sr, data = read(buf)
    return data, sr


a1=input('Enter 1, if you want to average the volume, enter 2, if you want to change the volume')
while a1!=1 and a1!=2:
  a1=input('Try again')
if a1==1:
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
  
          # === Выбор метода вычисления mean/std ===
      if method == 'global':
          mean_val = np.nanmean(S) if ignore_nan else np.mean(S)
          std_val = np.nanstd(S, ddof=ddof) if ignore_nan else np.std(S, ddof=ddof)
          mean_arr = mean_val
          std_arr = std_val
      elif method == 'per_freq':
          mean_arr = np.nanmean(S, axis=1, keepdims=True) if ignore_nan else np.mean(S, axis=1, keepdims=True)
          std_arr = np.nanstd(S, axis=1, ddof=ddof, keepdims=True) if ignore_nan else np.std(S, axis=1, ddof=ddof, keepdims=True)
      elif method == 'per_frame':
          mean_arr = np.nanmean(S, axis=0, keepdims=True) if ignore_nan else np.mean(S, axis=0, keepdims=True)
          std_arr = np.nanstd(S, axis=0, ddof=ddof, keepdims=True) if ignore_nan else np.std(S, axis=0, ddof=ddof, keepdims=True)
      else:
          print(f"Ошибка: method='{method}' не поддерживается. Используйте 'global', 'per_freq', 'per_frame'.")
          return Zxx, None, None  # вместо raise
  
      # === Определение маски для замены ===
      if mode == 'outlier':
          mask = np.abs(S - mean_arr) > std_arr
      elif mode == 'inside':
          mask = np.abs(S - mean_arr) < std_arr
      else:
          print(f"Ошибка: mode='{mode}' не поддерживается. Используйте 'outlier' или 'inside'.")
          return Zxx, None, None  # вместо raise
  
      # === Плавная замена (blending) ===
      S_mod = np.where(mask, (1.0 - blend) * S + blend * mean_arr, S)
  
  
      Zxx_mod = S_mod * np.exp(1j * phase) # 
      return Zxx_mod, np.squeeze(mean_arr), np.squeeze(std_arr)
  
  def process_file(infile, outfile,
                   method='per_freq',
                   mode='outlier',
                   blend=1.0,
                   nperseg=2048,
                   noverlap=1024,
                   window='hann',
                   ddof=0,
                   ignore_nan=True,
                   normalize=True):
      """
      Обработка аудиофайла: STFT → выравнивание амплитуд → ISTFT → сохранение.
      
      Возвращает:
      - stats: (mean, std) по каналам
      - f, t: частоты и время из STFT
      - S_orig, S_mod: амплитуды до и после
      """
      try:
          data, sr = sf.read(infile, dtype='float32')
      except Exception as e:
          print(f"Ошибка чтения файла: {e}")
          return None, None, None, None, None
  
      if data.ndim == 1:
          data = data[:, None]  # (n_samples,) → (n_samples, 1)
  
      n_samples, n_channels = data.shape
      out = np.zeros_like(data, dtype='float32')
  
      stats = []
      saved_f = saved_t = S_orig = S_mod = None
  
      for ch in range(n_channels):
          y = data[:, ch]
  
          # Прямое преобразование
          f, t, Zxx = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap, window=window)
          S_orig = np.abs(Zxx)  # Сохраняем оригинал
  
          # Выравнивание
          Zxx_mod, mean, std = equalize_amplitudes_Zxx(
              Zxx, method=method, mode=mode, blend=blend, ddof=ddof, ignore_nan=ignore_nan
          )
          if Zxx_mod is None:  # ошибка в equalize
              return None, None, None, None, None
  
          S_mod = np.abs(Zxx_mod)  # Сохраняем после обработки
        # Обратное преобразование
        _, y_mod = istft(Zxx_mod, fs=sr, nperseg=nperseg, noverlap=noverlap, window=window)

        # Подгонка длины
        if len(y_mod) > n_samples:
            y_mod = y_mod[:n_samples]
        elif len(y_mod) < n_samples:
            y_mod = np.pad(y_mod, (0, n_samples - len(y_mod)))

        out[:, ch] = y_mod.astype('float32')
        stats.append((mean, std))

        if ch == 0:
            saved_f, saved_t = f, t

    # Нормализация
    if normalize:
        maxv = np.max(np.abs(out))
        if maxv > 1e-8:  # избегаем деления на 0
            out = out / maxv

    # Моно
    if n_channels == 1:
        out = out[:, 0]

    # Сохранение
    try:
        sf.write(outfile, out, sr)
        print(f"Обработанный файл сохранён: {outfile}")
    except Exception as e:
        print(f"Ошибка записи файла: {e}")
        return None, None, None, None, None

    return stats, saved_f, saved_t, S_orig, S_mod

  # 1. Запись
  print("Нажмите кнопку и говорите...")
  y, sr = get_audio()
  sf.write("input.wav", y, sr)
  print(f"Записано: {len(y):,} сэмплов, частота: {sr} Гц")
  display(Audio(y, rate=sr))
  
  # 2. Обработка
  result = process_file(
      infile="input.wav",
      outfile="output_processed.wav",
      method='per_freq',     # выравнивание по каждой частоте отдельно
      mode='outlier',        # выравниваем выбросы (вне mean ± std)
      blend=1.0,             # полная замена на среднее
      nperseg=2048,
      noverlap=1024,
      normalize=True
  )
  
  if result is None:
      print("Ошибка при обработке.")
  else:
      stats, f, t, S_orig, S_mod = result
      mean0, std0 = stats[0]  # данные первого канала
  
  
  
  # Спектрограммы: до и после
      S_db = 20 * np.log10(S_orig + 1e-10)
      S_mod_db = 20 * np.log10(S_mod + 1e-10)
  
      plt.figure(figsize=(14, 5))
  
      plt.subplot(1, 2, 1)
      plt.pcolormesh(t, f, S_db, shading='gouraud', cmap='plasma', vmin=S_db.min(), vmax=S_db.max())
      plt.colorbar(label='Амплитуда (дБ)')
      plt.ylabel('Частота (Гц)')
      plt.xlabel('Время (с)')
      plt.title('Спектрограмма (до)')
      plt.yscale('log')
  
      plt.subplot(1, 2, 2)
      plt.pcolormesh(t, f, S_mod_db, shading='gouraud', cmap='plasma', vmin=S_mod_db.min(), vmax=S_mod_db.max())
      plt.colorbar(label='Амплитуда (дБ)')
      plt.ylabel('Частота (Гц)')
      plt.xlabel('Время (с)')
      plt.title('Спектрограмма (после)')
      plt.yscale('log')
  
      plt.tight_layout()
      plt.show()
  
  
  # амплитуда по частотам
      frame_idx = S_orig.shape[1] // 2
      amp_before = S_orig[:, frame_idx]
      amp_after = S_mod[:, frame_idx]
      mean_line = mean0 if mean0.ndim == 1 else mean0[:, np.newaxis]
      mean_line = mean_line.flatten()
  
      db_before = 20 * np.log10(amp_before + 1e-10)
      db_after = 20 * np.log10(amp_after + 1e-10)
      db_mean = 20 * np.log10(mean_line + 1e-10)
  
      fig = go.Figure()
      fig.add_trace(go.Scatter(x=f, y=db_before, mode='lines', name='До', line=dict(color='blue')))
      fig.add_trace(go.Scatter(x=f, y=db_after, mode='lines', name='После', line=dict(color='red', dash='dot')))
      fig.add_trace(go.Scatter(x=f, y=db_mean, mode='lines', name='Средняя амплитуда', line=dict(color='green', dash='dash')))
  
      fig.update_layout(
          title="Амплитуда по частотам (в дБ) — один кадр",
          xaxis_title="Частота (Гц)",
          yaxis_title="Амплитуда (дБ)",
          xaxis=dict(type="log"),
          template="plotly_white",
          width=800,
          height=500
      )
      fig.show()
  
  
  # 5. Воспроизведение
      y_out, sr_out = sf.read("output_processed.wav")
      print("Оригинал:")
      display(Audio(y, rate=sr))
      print("Обработанный:")
      display(Audio(y_out, rate=sr_out))
elif a1==2:
  # пример использования
  audio, sr = get_audio(duration=5)
  display(Audio(audio, rate=sr))
  
  # спрашиваем у пользователя коэффициент громкости
  factor = float(input("Введите коэффициент громкости (например, 0.5 для уменьшения, 2 для увеличения): "))
  
  # изменяем громкость
  audio_adjusted = audio * factor
  
  # отсечение значений, чтобы не выходили за [-1, 1]
  audio_adjusted = audio_adjusted.clip(-1, 1)
  
  # сохраняем результат
  sf.write("output_adjusted.wav", audio_adjusted, sr)
  display(Audio(audio_adjusted, rate=sr))
