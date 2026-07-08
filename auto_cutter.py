import os
import argparse
import pandas as pd
import numpy as np
from moviepy.editor import VideoFileClip, VideoClip
from PIL import Image, ImageDraw, ImageFont

def load_rapsodo_data(csv_path, first_pitch_time):
    # 重複する列名があ记录されても自動でリネームして読み込む
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # 1列目の「日付」または「Date」列を最優先で使用する
    time_col = None
    for col in df.columns:
        if '日付' in str(col) or str(col).strip().lower() == 'date':
            # ちゃんと日付っぽい文字列が入っているか確認
            sample_val = df[col].dropna().astype(str).values
            if len(sample_val) > 0 and any('/' in s or '-' in s for s in sample_val[:5]):
                time_col = col
                break
                
    if time_col is None:
        # 万が一見つからなければ、最初に見つかった「time」や「日付」を含む列を使用
        time_keywords = ['日付', 'date', 'time', '時刻']
        for col in df.columns:
            if any(kw in str(col).lower() for kw in time_keywords):
                time_col = col
                break

    if time_col is None:
        print("警告: 日付列が特定できなかったため、15秒間隔で自動配置します。")
        df['Elapsed_Seconds'] = df.index * 15.0
    else:
        # スラッシュやハイフン区切りの日付（2026/05/27など）を綺麗に変換
        df['Timestamp'] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)
        start_time = df['Timestamp'].iloc[0]
        df['Elapsed_Seconds'] = (df['Timestamp'] - start_time).dt.total_seconds()
    
    df['Video_Target_Time'] = first_pitch_time + df['Elapsed_Seconds']
    return df

def get_column_value(row, keywords, default="0"):
    for col in row.index:
        if any(kw.lower() in str(col).lower() for kw in keywords):
            val = str(row[col]).strip()
            if val and val != 'nan':
                return val
    return default

def make_hud_layer(width, height, row):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("arial.ttf", 20)
        font_main = ImageFont.truetype("arial.ttf", 45)
        font_sub = ImageFont.truetype("arial.ttf", 28)
    except IOError:
        font_title = font_main = font_sub = ImageFont.load_default()

    speed_val = get_column_value(row, ['pitchballvelocity', 'velocity', 'speed', '球速', 'ball speed'])
    try: speed = f"{float(speed_val):.1f} MPH"
    except: speed = f"{speed_val} MPH"

    v_brk = get_column_value(row, ['verticalbreak', 'vb ', '縦変化', 'inducedvertical', 'vb (']) + " IN"
    h_brk = get_column_value(row, ['horizontalbreak', 'hb ', '横変化', 'hb (']) + " IN"
    p_type = get_column_value(row, ['pitchtype', 'pitch type', '球種', 'type'], default="UNKNOWN")

    draw.rounded_rectangle([40, 40, 420, 290], radius=12, fill=(0, 60, 100, 50), outline=(0, 180, 255, 200), width=2)
    draw.text((55, 50), "SEET BT - RAPSODO", fill=(255, 255, 255, 200), font=font_title)
    draw.line([(55, 75), (405, 75)], fill=(0, 180, 255, 150), width=1)
    draw.text((55, 90), f"SPEED: {speed}", fill=(0, 255, 150, 255), font=font_main)
    draw.text((55, 155), f"V-BRK: {v_brk}", fill=(255, 255, 255, 220), font=font_sub)
    draw.text((55, 195), f"H-BRK: {h_brk}", fill=(255, 255, 255, 220), font=font_sub)
    draw.text((55, 240), f"TYPE: {p_type}", fill=(255, 200, 0, 255), font=font_sub)
    return np.array(img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--time', type=float, required=True)
    args = parser.parse_args()

    df = load_rapsodo_data(args.csv, args.time)
    video = VideoFileClip(args.video)
    width, height = video.size
    clips = []
    
    duration = 5.0
    offset = 2.0

    for idx, row in df.iterrows():
        target_time = row['Video_Target_Time']
        start_cut = target_time - offset
        end_cut = start_cut + duration
        if end_cut > video.duration: break
            
        sub_clip = video.subclip(start_cut, end_cut)
        hud_np = make_hud_layer(width, height, row)
        rgb_layer = hud_np[:, :, :3]
        alpha_layer = hud_np[:, :, 3] / 255.0
        
        combined_clip = VideoClip(
            lambda t: (sub_clip.get_frame(t) * (1 - alpha_layer[:,:,None]) + rgb_layer * alpha_layer[:,:,None]).astype('uint8'),
            duration=duration
        )
        if sub_clip.audio: combined_clip = combined_clip.set_audio(sub_clip.audio)
        clips.append(combined_clip)
    
    if len(clips) == 0:
        print("エラー: 切り出せる投球がありませんでした。動画の長さや初球の時間を確認してください。")
        exit(3)

    from moviepy.editor import concatenate_videoclips
    final_video = concatenate_videoclips(clips)
    final_video.write_videofile("highlight_output.mp4", fps=video.fps, codec="libx264", audio_codec="aac")
    video.close()
    final_video.close()

if __name__ == "__main__":
    main()
