import yt_dlp
import os
from typing import List, Dict
import config

class VideoCollector:
    def __init__(self, output_dir: str = config.DATA_DIR):
        """Initialize video collector with output directory
        
        Args:
            output_dir: Directory to save downloaded videos
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    def download_videos(self, video_urls: List[str]) -> List[str]:
        """Download videos from YouTube URLs
        
        Args:
            video_urls: List of YouTube video URLs to download
            
        Returns:
            List of paths to downloaded video files
        """
        video_paths = []
        
        for i, url in enumerate(video_urls):
            # Create unique filename for each video
            output_path = os.path.join(self.output_dir, f'video_{i:02d}.mp4')
            
            if os.path.exists(output_path):
                print(f"Video {i+1} already exists, skipping download")
                video_paths.append(output_path)
                continue
            
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'outtmpl': output_path,  # Use our custom path
                'quiet': False,
                'no_warnings': True
            }
            
            try:
                print(f"Downloading video {i+1}/{len(video_urls)}")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                    video_paths.append(output_path)
                print(f"Successfully downloaded: {output_path}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")
        
        return video_paths