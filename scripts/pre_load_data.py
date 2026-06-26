import shutil
import re
from pathlib import Path

def organize_raw_data(raw_dir: str = "data/raw"):
    base_path = Path(raw_dir)
    
    # 兼容前缀逻辑：[\d\.]* 表示匹配0个或多个数字与小数点
    pattern = re.compile(r'(.*cricket_\d+_\d+_\d+_session_\d+)')

    for csv_file in base_path.rglob("*.csv"):
        match = pattern.search(csv_file.name)
        if match:
            # 提取结果示例: 
            # "0.828cricket_001_20260624_231006_session_1"
            # 或 "cricket_001_20260617_223444_session_1"
            session_folder_name = match.group(1)
            target_dir = base_path / session_folder_name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = target_dir / csv_file.name
            
            # 避免同目录移动报错
            if csv_file.parent != target_dir:
                shutil.move(str(csv_file), str(target_path))
                print(f"归档: {csv_file.name} -> {session_folder_name}/")

if __name__ == "__main__":
    organize_raw_data()