#!/usr/bin/env python3
"""
Скрипт для удаления _legacy из всех импортов в папке nn.tracker
"""

import os
import re
from pathlib import Path

def find_tracker_files(tracker_dir):
    """Находит все Python файлы в папке tracker"""
    python_files = []
    
    for root, dirs, files in os.walk(tracker_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
    
    return python_files

def fix_legacy_imports_in_file(file_path):
    """Исправляет legacy импорты в конкретном файле"""
    print(f"Processing: {file_path}")
    
    # Читаем файл
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return False
    
    original_content = content
    
    # Паттерны для исправления
    patterns_to_fix = [
        # Импорты с _legacy
        (r'from supervisely\.nn\.tracker\.tracker_legacy import', 'from supervisely.nn.tracker.tracker import'),
        (r'import supervisely\.nn\.tracker\.tracker_legacy', 'import supervisely.nn.tracker.tracker'),
        
        # Импорты bot_sort_legacy
        (r'from supervisely\.nn\.tracker\.bot_sort_legacy', 'from supervisely.nn.tracker.bot_sort_legacy'),  # Оставляем bot_sort_legacy как есть
        (r'import supervisely\.nn\.tracker\.bot_sort_legacy', 'import supervisely.nn.tracker.bot_sort_legacy'),  # Оставляем bot_sort_legacy как есть
        
        # Общие паттерны с _legacy в путях
        (r'supervisely\.nn\.tracker\.([^.]+)_legacy', r'supervisely.nn.tracker.\1'),
        
        # Исправляем дублированные supervisely.supervisely обратно к нормальному виду
        (r'supervisely\.supervisely\.nn\.tracker', 'supervisely.nn.tracker'),
        
        # Конкретные известные проблемные импорты
        (r'from supervisely\.supervisely\.nn\.tracker\.tracker import', 'from supervisely.nn.tracker.tracker import'),
        (r'import supervisely\.supervisely\.nn\.tracker\.tracker', 'import supervisely.nn.tracker.tracker'),
    ]
    
    changes_made = False
    changes_log = []
    
    for pattern, replacement in patterns_to_fix:
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, replacement, content)
            changes_made = True
            changes_log.append(f"    Fixed pattern: {pattern} -> {replacement}")
    
    # Специальная обработка для bot_sort_legacy - НЕ убираем _legacy из названия папки
    # Но убираем _legacy из других мест где он не нужен
    
    if changes_made:
        # Создаем резервную копию
        backup_path = file_path + '.backup'
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            print(f"  💾 Created backup: {backup_path}")
        except Exception as e:
            print(f"  ⚠️ Warning: Could not create backup: {e}")
        
        # Записываем исправленный файл
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ Fixed imports in file")
            for change in changes_log:
                print(change)
            return True
        except Exception as e:
            print(f"  ❌ Error writing file: {e}")
            return False
    else:
        print(f"  ✅ No legacy imports found")
        return False

def fix_specific_known_issues(tracker_dir):
    """Исправляет конкретные известные проблемы"""
    
    # Известные проблемные файлы
    problematic_files = [
        "bot_sort_legacy/sly_tracker.py",
        "botsort/sly_tracker.py", 
        "__init__.py"
    ]
    
    for rel_path in problematic_files:
        file_path = os.path.join(tracker_dir, rel_path)
        if os.path.exists(file_path):
            print(f"\n🔧 Fixing known issue in: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Специфичные исправления
                if "sly_tracker.py" in rel_path:
                    # Исправляем конкретный проблемный импорт
                    content = content.replace(
                        'from supervisely.supervisely.nn.tracker.tracker import BaseDetection as Detection',
                        'from supervisely.nn.tracker.tracker import BaseDetection as Detection'
                    )
                    content = content.replace(
                        'from supervisely.supervisely.nn.tracker.tracker import BaseTrack, BaseTracker',
                        'from supervisely.nn.tracker.tracker import BaseTrack, BaseTracker'
                    )
                
                if "__init__.py" in rel_path:
                    # Исправляем импорты в __init__.py
                    content = re.sub(
                        r'from supervisely\.nn\.tracker\.([^.]+)_legacy import',
                        r'from supervisely.nn.tracker.\1_legacy import',  # Для bot_sort_legacy оставляем как есть
                        content
                    )
                
                # Записываем исправленный файл
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✅ Fixed specific issues")
                
            except Exception as e:
                print(f"  ❌ Error fixing {file_path}: {e}")

def test_imports_after_fix():
    """Тестирует импорты после исправления"""
    print("\n" + "="*60)
    print("🧪 TESTING IMPORTS AFTER FIX")
    print("="*60)
    
    try:
        import supervisely as sly
        print("✅ import supervisely - OK")
        
        from supervisely.nn.tracker.tracker import BaseDetection, BaseTracker
        print("✅ import BaseDetection, BaseTracker - OK")
        
        try:
            from supervisely.nn.tracker.botsort.tracker.mc_bot_sort import BoTSORT
            print("✅ import BoTSORT (new) - OK")
        except ImportError as e:
            print(f"❌ import BoTSORT (new) failed: {e}")
        
        try:
            from supervisely.nn.tracker.botsort.sly_tracker import BoTTracker
            print("✅ import BoTTracker (new) - OK")
        except ImportError as e:
            print(f"❌ import BoTTracker (new) failed: {e}")
            
        try:
            from supervisely.nn.tracker.bot_sort_legacy.sly_tracker import BoTTracker as LegacyBoTTracker
            print("✅ import BoTTracker (legacy) - OK")
        except ImportError as e:
            print(f"❌ import BoTTracker (legacy) failed: {e}")
        
        try:
            # Проверяем основной импорт из __init__.py
            from supervisely.nn.tracker import BoTTracker
            print("✅ import BoTTracker from nn.tracker - OK")
        except ImportError as e:
            print(f"❌ import BoTTracker from nn.tracker failed: {e}")
        
        print("🎉 Import testing completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Basic supervisely import failed: {e}")
        return False

def main():
    print("🔧 REMOVING _LEGACY FROM TRACKER IMPORTS")
    print("="*60)
    
    # Путь к папке tracker
    tracker_dir = "/root/tracking-by-detection/supervisely/supervisely/nn/tracker"
    
    if not os.path.exists(tracker_dir):
        print(f"❌ Tracker directory not found: {tracker_dir}")
        return
    
    print(f"📁 Working in: {tracker_dir}")
    
    # 1. Находим все Python файлы
    python_files = find_tracker_files(tracker_dir)
    print(f"📄 Found {len(python_files)} Python files")
    
    # 2. Исправляем legacy импорты в каждом файле
    print("\n" + "-"*50)
    print("FIXING LEGACY IMPORTS IN FILES:")
    print("-"*50)
    
    total_fixed = 0
    for file_path in python_files:
        if fix_legacy_imports_in_file(file_path):
            total_fixed += 1
    
    print(f"\n📊 Fixed legacy imports in {total_fixed} files")
    
    # 3. Исправляем конкретные известные проблемы
    print("\n" + "-"*50)
    print("FIXING SPECIFIC KNOWN ISSUES:")
    print("-"*50)
    
    fix_specific_known_issues(tracker_dir)
    
    # 4. Тестируем импорты
    test_imports_after_fix()
    
    print("\n" + "="*60)
    print("🎯 SUMMARY:")
    print(f"✅ Processed {len(python_files)} files")
    print(f"✅ Fixed legacy imports in {total_fixed} files")
    print("✅ Applied specific fixes for known issues")
    print("\nYou can now try running your tracking script again!")
    print("="*60)

if __name__ == "__main__":
    main()