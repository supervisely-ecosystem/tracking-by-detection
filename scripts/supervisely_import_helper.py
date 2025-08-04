
import os
import sys
from pathlib import Path

def setup_supervisely_imports():
    """
    Настройка путей для корректного импорта supervisely модулей
    Вызывайте эту функцию в начале ваших скриптов
    """
    # Определяем возможные пути к supervisely
    current_dir = Path.cwd()
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    
    possible_paths = [
        current_dir / "supervisely",
        script_dir / "supervisely",
        script_dir.parent / "supervisely", 
        current_dir.parent / "supervisely"
    ]
    
    sly_root = None
    for path in possible_paths:
        if path.exists() and (path / "supervisely").exists():
            sly_root = path
            break
    
    if sly_root and str(sly_root) not in sys.path:
        sys.path.insert(0, str(sly_root))
        print(f"Added supervisely path: {sly_root}")
        return sly_root
    
    return None

def safe_import_trackers():
    """
    Безопасный импорт трекеров с обработкой ошибок
    """
    trackers = {}
    
    try:
        from supervisely.nn.tracker.botsort.tracker.mc_bot_sort import BoTSORT
        trackers['BoTSORT_ORIG'] = BoTSORT
        print("✅ Original BoTSORT imported successfully")
    except ImportError as e:
        print(f"⚠️  Original BoTSORT import failed: {e}")
        trackers['BoTSORT_ORIG'] = None
    
    try:
        from supervisely.nn.tracker.botsort.sly_tracker import BoTTracker
        trackers['BoTTracker'] = BoTTracker
        print("✅ Supervisely BoTTracker imported successfully")
    except ImportError as e:
        print(f"⚠️  Supervisely BoTTracker import failed: {e}")
        trackers['BoTTracker'] = None
    
    return trackers

if __name__ == "__main__":
    setup_supervisely_imports()
