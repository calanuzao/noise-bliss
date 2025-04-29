"""

The wrapper (SonycUST class) provides a standardized interface to:

Load audio files
Access annotations
Manage dataset metadata
Validate data integrity

"""

from pathlib import Path
import pandas as pd
import soundata

class SonycUST:
    """SONYC Urban Sound Tagging dataset wrapper"""
    
    def __init__(self, data_home=None):
        self.data_home = Path(data_home) if data_home else Path("data/sonyc-ust")
        self.annotations_path = self.data_home / "annotations.csv"
        self.audio_dirs = sorted(list(self.data_home.glob("audio-*")))
        self.label_taxonomy = {
            'engine': 0,
            'machinery-impact': 1,
            'non-machinery-impact': 2,
            'powered-saw': 3,
            'alert-signal': 4,
            'music': 5,
            'human-voice': 6,
            'dog': 7
        }
        
        if not self.audio_dirs:
            raise RuntimeError("No audio directories found")
        if not self.annotations_path.exists():
            raise RuntimeError(f"Annotations file not found at {self.annotations_path}")

    def load_annotations(self):
        """Load annotations file"""
        return pd.read_csv(self.annotations_path)

    def get_audio_path(self, audio_filename):
        """Get full path for an audio file by searching in all audio directories"""
        for audio_dir in self.audio_dirs:
            file_path = audio_dir / audio_filename
            if file_path.exists():
                return file_path
        return None

    def validate_structure(self):
        """Validate dataset structure"""
        print(f"Found {len(self.audio_dirs)} audio directories")
        print(f"Annotations file exists: {self.annotations_path.exists()}")
        
        # Sample audio file check
        annotations = self.load_annotations()
        if len(annotations) > 0:
            sample_file = annotations['audio_filename'].iloc[0]
            sample_path = self.get_audio_path(sample_file)
            print(f"Sample audio file found: {sample_path is not None}")
        
        return True