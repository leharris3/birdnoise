"""
Package only essential files needed for training reproduction.

Creates a zip file containing:
- Training scripts (excluding debug/test scripts)
- NatureLM-audio code
- Requirements/dependencies
- README/documentation
- Configuration files

Excludes:
- Data files
- Checkpoints
- Wandb logs
- Python cache
- Large assets
- Temporary files
"""

import zipfile
import os
from pathlib import Path
from datetime import datetime


def should_include_file(file_path: Path, root_dir: Path) -> bool:
    """Determine if a file should be included in the package."""
    rel_path = file_path.relative_to(root_dir)
    
    # Always exclude
    exclude_patterns = [
        # Data and checkpoints
        'Data',
        'checkpoints',
        'checkpoints_beans',
        'checkpoints_fusion',
        'checkpoints_test',
        'evaluation_results',
        
        # Logs and cache
        'wandb',
        '__pycache__',
        '.pyc',
        '.pyd',
        '.pyo',
        '.egg-info',
        '*.log',
        '.git',
        '.vscode',
        '.idea',
        
        # Large or unnecessary files
        '*.tar.gz',
        '*.zip',
        '*.mp3',
        '*.wav',
        '*.h5',
        '*.tif',
        '*.png',
        '*.jpg',
        '*.jpeg',
        '*.pdf',
        
        # Temporary and OS files
        '.DS_Store',
        'Thumbs.db',
        '*.tmp',
        '*.temp',
        '*.swp',
        '*.swo',
        '*~',
        
        # Test and debug scripts (optional - comment out if you want them)
        'test_*.py',
        'check_*.py',
        'debug_*.py',
        'quick_test_*.py',
        '*.ipynb',  # Jupyter notebooks
        'profile_*.py',
        
        # Specific files to exclude
        'oovanger0812@ls6.tacc.utexas.edu',
        'hpc_training_files*.zip',
        'BirdNoise.tar.gz',
        
        # Large assets in NatureLM
        'NatureLM-audio/assets/*.mp3',
        'NatureLM-audio/assets/*.wav',
    ]
    
    # Check against exclude patterns
    path_str = str(rel_path)
    for pattern in exclude_patterns:
        if pattern.startswith('*'):
            if path_str.endswith(pattern[1:]):
                return False
        elif pattern.endswith('*'):
            if path_str.startswith(pattern[:-1]):
                return False
        else:
            if pattern in path_str or path_str.startswith(pattern + '/'):
                return False
    
    # Include Python files, configs, README, requirements
    include_extensions = ['.py', '.yml', '.yaml', '.json', '.txt', '.md', 
                          '.toml', '.lock', '.sh', '.cfg', '.ini', '.toml']
    
    if file_path.suffix in include_extensions:
        return True
    
    # Include directories that might be needed
    if file_path.is_dir():
        # Include important directories even if they have no matching files yet
        dir_name = file_path.name
        if dir_name in ['NatureLM', 'configs', 'Scripts']:
            return True
    
    return False


def should_include_dir(dir_path: Path, root_dir: Path) -> bool:
    """Determine if a directory should be traversed."""
    rel_path = dir_path.relative_to(root_dir)
    path_str = str(rel_path)
    
    # Don't traverse excluded directories
    exclude_dirs = [
        'Data', 'checkpoints', 'wandb', '__pycache__', '.git',
        'checkpoints_beans', 'checkpoints_fusion', 'checkpoints_test',
        'evaluation_results', '.vscode', '.idea', 'assets'
    ]
    
    for exclude in exclude_dirs:
        if path_str.startswith(exclude):
            return False
    
    return True


def create_package(output_path: Path, root_dir: Path = None):
    """Create a zip package with only essential files."""
    if root_dir is None:
        root_dir = Path(__file__).parent.parent  # Project root
    
    output_path = Path(output_path)
    if output_path.suffix != '.zip':
        output_path = output_path.with_suffix('.zip')
    
    print(f"Creating reproduction package: {output_path}")
    print(f"Root directory: {root_dir}")
    print()
    
    included_files = []
    excluded_files = []
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add files
        for root, dirs, files in os.walk(root_dir):
            root_path = Path(root)
            
            # Filter directories to traverse
            dirs[:] = [d for d in dirs if should_include_dir(root_path / d, root_dir)]
            
            for file in files:
                file_path = root_path / file
                
                if should_include_file(file_path, root_dir):
                    rel_path = file_path.relative_to(root_dir)
                    zipf.write(file_path, rel_path)
                    included_files.append(rel_path)
                else:
                    rel_path = file_path.relative_to(root_dir)
                    excluded_files.append(rel_path)
    
    print(f"\n{'='*60}")
    print(f"Package created: {output_path}")
    print(f"{'='*60}")
    print(f"Included files: {len(included_files)}")
    print(f"Excluded files: {len(excluded_files)}")
    print(f"\nPackage size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    print(f"\n{'='*60}")
    print("Top-level structure included:")
    print(f"{'='*60}")
    
    # Show structure
    top_level = set()
    for file in included_files:
        parts = file.parts
        if len(parts) > 0:
            top_level.add(parts[0])
    
    for item in sorted(top_level):
        count = sum(1 for f in included_files if f.parts[0] == item)
        print(f"  {item}/ ({count} files)")
    
    print(f"\n{'='*60}")
    print("Key files included:")
    print(f"{'='*60}")
    
    key_files = [f for f in included_files if Path(f).name in [
        'requirements.txt', 'README.md', 'pyproject.toml', 
        'train_weighted_fusion.py', 'train_lc.py', 'evaluate_models.py',
        'eBirdPrior.py', 'precompute_priors.py'
    ]]
    
    for key_file in sorted(key_files):
        print(f"  {key_file}")
    
    print(f"\n{'='*60}")
    print("Instructions for reproduction:")
    print(f"{'='*60}")
    print("1. Extract the zip file")
    print("2. Install dependencies: pip install -r NatureLM-audio/requirements.txt")
    print("3. Install additional packages: pip install scikit-learn seaborn h5py rasterio")
    print("4. Set up HuggingFace authentication for NatureLM-audio")
    print("5. Prepare data directory with CBI dataset")
    print("6. Run training: python Scripts/train_weighted_fusion.py ...")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Package essential files for reproduction")
    parser.add_argument("--output", type=str, default=None,
                       help="Output zip file path (default: BirdNoise_reproduction_YYYYMMDD.zip)")
    parser.add_argument("--root", type=str, default=None,
                       help="Root directory to package (default: parent of Scripts)")
    
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        script_dir = Path(__file__).parent
        args.output = script_dir.parent / f"BirdNoise_reproduction_{timestamp}.zip"
    
    root_dir = Path(args.root) if args.root else Path(__file__).parent.parent
    
    create_package(args.output, root_dir)
