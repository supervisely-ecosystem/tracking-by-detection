#!/usr/bin/env python3
"""
Скрипт для диагностики проблем с импортами и исправления путей
"""

import os
import sys
from pathlib import Path

def debug_project_structure():
    """Анализ структуры проекта"""
    print("=== PROJECT STRUCTURE ANALYSIS ===")
    
    # Текущая директория
    current_dir = Path.cwd()
    script_dir = Path(__file__).parent
    
    print(f"Current working directory: {current_dir}")
    print(f"Script directory: {script_dir}")
    print(f"Python path: {sys.path[:3]}...")
    
    # Поиск папки supervisely
    possible_sly_paths = [
        current_dir / "supervisely",
        script_dir / "supervisely", 
        script_dir.parent / "supervisely",
        current_dir.parent / "supervisely"
    ]
    
    print("\n=== LOOKING FOR SUPERVISELY DIRECTORY ===")
    sly_path = None
    for path in possible_sly_paths:
        print(f"Checking: {path}")
        if path.exists() and path.is_dir():
            print(f"  ✅ Found!")
            if (path / "supervisely").exists():
                print(f"  ✅ Contains supervisely subdirectory")
                sly_path = path
                break
            elif (path / "__init__.py").exists():
                print(f"  ✅ Is a Python package")
                sly_path = path.parent
                break
        else:
            print(f"  ❌ Not found")
    
    if sly_path:
        print(f"\n✅ Supervisely path found: {sly_path}")
        return sly_path
    else:
        print(f"\n❌ Supervisely path not found!")
        return None

def test_basic_imports(sly_path):
    """Тестирование базовых импортов"""
    print("\n=== TESTING BASIC IMPORTS ===")
    
    if sly_path and sly_path not in sys.path:
        sys.path.insert(0, str(sly_path))
        print(f"Added to sys.path: {sly_path}")
    
    # Тест основного импорта
    try:
        import supervisely as sly
        print("✅ import supervisely as sly - OK")
        print(f"   Supervisely version: {getattr(sly, '__version__', 'unknown')}")
        print(f"   Supervisely location: {sly.__file__}")
    except ImportError as e:
        print(f"❌ import supervisely failed: {e}")
        return False
    
    # Тест импорта аннотаций
    try:
        from supervisely import Annotation, Label, Rectangle
        print("✅ from supervisely import Annotation, Label, Rectangle - OK")
    except ImportError as e:
        print(f"❌ Annotation imports failed: {e}")
        return False
    
    return True

def test_tracker_imports():
    """Тестирование импортов трекеров"""
    print("\n=== TESTING TRACKER IMPORTS ===")
    
    # Тест импорта tracker базового модуля
    try:
        from supervisely.nn.tracker import tracker
        print("✅ from supervisely.nn.tracker import tracker - OK")
        print(f"   Tracker module location: {tracker.__file__}")
    except ImportError as e:
        print(f"❌ Base tracker import failed: {e}")
    
    # Тест импорта BaseTracker
    try:
        from supervisely.nn.tracker.tracker import BaseTracker
        print("✅ from supervisely.nn.tracker.tracker import BaseTracker - OK")
    except ImportError as e:
        print(f"❌ BaseTracker import failed: {e}")
    
    # Тест импорта оригинального BoTSORT
    try:
        from supervisely.nn.tracker.botsort.tracker.mc_bot_sort import BoTSORT
        print("✅ from supervisely.nn.tracker.botsort.tracker.mc_bot_sort import BoTSORT - OK")
    except ImportError as e:
        print(f"❌ Original BoTSORT import failed: {e}")
        print(f"    Error details: {e}")
    
    # Тест импорта Supervisely обертки
    try:
        from supervisely.nn.tracker.botsort.sly_tracker import BoTTracker
        print("✅ from supervisely.nn.tracker.botsort.sly_tracker import BoTTracker - OK")
    except ImportError as e:
        print(f"❌ Supervisely BoTTracker import failed: {e}")
        print(f"    Error details: {e}")

def explore_tracker_directory(sly_path):
    """Исследование структуры директории трекеров"""
    print("\n=== EXPLORING TRACKER DIRECTORY STRUCTURE ===")
    
    if not sly_path:
        print("❌ No supervisely path provided")
        return
    
    tracker_path = sly_path / "supervisely" / "nn" / "tracker"
    print(f"Looking in: {tracker_path}")
    
    if not tracker_path.exists():
        print(f"❌ Tracker directory not found: {tracker_path}")
        return
    
    print("📁 Tracker directory contents:")
    for item in sorted(tracker_path.iterdir()):
        if item.is_dir():
            print(f"  📁 {item.name}/")
            # Показываем содержимое важных директорий
            if item.name in ['botsort', 'bot_sort_legacy']:
                for subitem in sorted(item.iterdir()):
                    if subitem.is_file() and subitem.suffix == '.py':
                        print(f"    📄 {subitem.name}")
                    elif subitem.is_dir():
                        print(f"    📁 {subitem.name}/")
        elif item.suffix == '.py':
            print(f"  📄 {item.name}")

def create_fixed_import_helper():
    """Создание помощника для исправления импортов"""
    print("\n=== CREATING IMPORT HELPER ===")
    
    helper_code = '''
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
'''
    
    helper_file = Path("supervisely_import_helper.py")
    with open(helper_file, 'w') as f:
        f.write(helper_code)
    
    print(f"✅ Created import helper: {helper_file.absolute()}")
    print("   Use: from supervisely_import_helper import setup_supervisely_imports, safe_import_trackers")

def main():
    print("🔍 SUPERVISELY IMPORT DIAGNOSTICS")
    print("=" * 50)
    
    # 1. Анализ структуры проекта
    sly_path = debug_project_structure()
    
    # 2. Тестирование базовых импортов
    if test_basic_imports(sly_path):
        # 3. Тестирование импортов трекеров
        test_tracker_imports()
        
        # 4. Исследование структуры директорий
        explore_tracker_directory(sly_path)
    
    # 5. Создание помощника для импортов
    create_fixed_import_helper()
    
    print("\n" + "=" * 50)
    print("🎯 RECOMMENDATIONS:")
    print("1. Use the created 'supervisely_import_helper.py' in your scripts")
    print("2. Check if supervisely directory structure matches expected layout")
    print("3. Verify that all __init__.py files exist in the supervisely package")

if __name__ == "__main__":
    main()