import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True, help='input dir with labels')
parser.add_argument('--output_file', required=True, help='path to save output union file')
args = parser.parse_args()

input_dir = args.input_dir
output_file = args.output_file
img_w, img_h = 1280, 720

os.makedirs(os.path.dirname(output_file), exist_ok=True)
frame_pattern = re.compile(r'-(\d+)\.txt$')
frama_pattern2 = re.compile(r'_(\d+)\.txt$')

with open(output_file, 'w') as out:
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith('.txt'): 
            continue
            
        m = frame_pattern.search(fname)
        if not m:
            m = frama_pattern2.search(fname)
            if not m:
                print(f'Пропускаю файл: {fname}')
                continue
        
        # Проверьте правильность этой формулы для вашего случая!
        frame_id = int(m.group(1))  # или int(m.group(1)) + 1
        
        seen_detections = set()
        
        with open(os.path.join(input_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Дедупликация
                if line in seen_detections:
                    print(f'Пропускаю дубликат в {fname}: {line}')
                    continue
                seen_detections.add(line)
                
                parts = line.split()
                if len(parts) not in [6, 7]:
                    print(f'Неподходящая строка {fname}: {parts}')
                    continue
                
                # Парсинг данных
                if len(parts) == 6:
                    # Формат: cls cx cy rw rh track_id
                    cls, cx, cy, rw, rh, track_id = map(float, parts)
                    score = 1.0  # Boxmot не выводит score, ставим 1.0
                elif len(parts) == 7:
                    # Формат: cls cx cy rw rh score track_id
                    cls, cx, cy, rw, rh, score, track_id = map(float, parts)
                else:
                    print(f'Неожиданное количество частей: {len(parts)}')
                    continue
                
                # Конвертация YOLO -> пиксели
                x1 = (cx - rw/2) * img_w
                y1 = (cy - rh/2) * img_h
                w_px = rw * img_w
                h_px = rh * img_h

                # КРИТИЧЕСКИ ВАЖНО: используем track_id из файла!
                out.write(f'{frame_id},{int(track_id)},{x1:.2f},{y1:.2f},{w_px:.2f},{h_px:.2f},{score:.2f},-1,-1,-1\n')
                
print(f"Обработка завершена. Результат сохранен в: {output_file}")