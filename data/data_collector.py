from pytube import YouTube
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
            try:
                print(f"Downloading video {i+1}/{len(video_urls)}")
                yt = YouTube(url)
                # Get first progressive stream (has both video & audio)
                video = yt.streams.filter(progressive=True, file_extension='mp4').first()
                filename = f'video_{i}.mp4'
                path = os.path.join(self.output_dir, filename)
                video.download(output_path=self.output_dir, filename=filename)
                video_paths.append(path)
                print(f"Successfully downloaded: {filename}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")
        
        return video_paths