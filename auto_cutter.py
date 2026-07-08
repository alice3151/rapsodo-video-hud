import os
import argparse
import pandas as pd
import numpy as np
from moviepy.editor import VideoFileClip, VideoClip
from PIL import Image, ImageDraw, ImageFont

def load_rapsodo_data(csv_path, first_pitch_time):
    df = pd.read_csv(csv_path)
    # ラプソードの仕様に合わせて列名を自動調整
    time_col = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()][0]
    df['Timestamp'] = pd.to_datetime(df[time_col]) 
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    start_time = df['Timestamp'].iloc[0]
    df['Elapsed_Seconds'] = (df['Timestamp'] - start_time).dt.total_seconds()
    df['Video_Target_Time'] = first_pitch_time + df['Elapsed_Seconds']
    return df

def make_hud_layer(width, height, row):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("arial.ttf", 20)
        font_main = ImageFont.truetype("arial.ttf", 45)
        font_sub = ImageFont.truetype("arial.ttf", 28)
    except IOError:
        font_title = font_main = font_sub = ImageFont.load_default()

    # 列名の揺れに対応
    speed = f"{row.get('Velocity', row.get('PitchBallSpeed', '0'))} MPH"
    v_break = f"{row.get('VerticalBreak', row.get('InducedVerticalBreak', '0'))} IN"
    h_break = f"{row.get('HorizontalBreak', row.get('HorizontalBreak', '0'))} IN"
    p_type = str(row.get('PitchType', 'UNKNOWN'))

    draw.rounded_rectangle([40, 40, 420, 290], radius=12, fill=(0, 60, 100, 50), outline=(0, 180, 255, 200), width=2)
    draw.text((55, 50), "SEET BT - RAPSODO", fill=(255, 255, 255, 200), font=font_title)
    draw.line([(55, 75), (405, 75)], fill=(0, 180, 255, 150), width=1)
    draw.text((55, 90), f"SPEED: {speed}", fill=(0, 255, 150, 255), font=font_main)
    draw.text((55, 155), f"V-BRK: {v_break}", fill=(255, 255, 255, 220), font=font_sub)
    draw.text((55, 195), f"H-BRK: {h_break}", fill=(255, 255, 255, 220), font=font_sub)
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
    
    from moviepy.editor import concatenate_videoclips
    final_video = concatenate_videoclips(clips)
    final_video.write_videofile("highlight_output.mp4", fps=video.fps, codec="libx264", audio_codec="aac")
    video.close()
    final_video.close()

if __name__ == "__main__":
    main()
